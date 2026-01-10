import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { vertexShader, fragmentShader } from './shaders';
import { Camera, RefreshCcw, Upload, Settings, AlertCircle, Hand, Video, VideoOff } from 'lucide-react';
// Import MediaPipe safely handling ESM export variations
import * as mpHandsPkg from '@mediapipe/hands';
import type { Results } from '@mediapipe/hands';
import * as mpCameraPkg from '@mediapipe/camera_utils';

// Extract Hands class from the package import (handling both default and named export scenarios)
const Hands = (mpHandsPkg as any).Hands || (mpHandsPkg as any).default?.Hands;
// Extract Camera class manually from camera_utils package
const MPCamera = (mpCameraPkg as any).Camera || (mpCameraPkg as any).default?.Camera;

// Use a simple relative path. The file should be in the public root.
const DEFAULT_IMAGE = "/A.png";
const WORLD_B_IMAGE = "/B.png";
const WORLD_C_MODEL = "/C.glb";
const MIN_ZOOM_DISTANCE = 50;
const MAX_ZOOM_DISTANCE = 2000;
const DEEP_ZONE_DISTANCE = 70;
const RETURN_ZONE_DISTANCE = 220;
const RETURN_MARGIN = 60;
const RETURN_ARM_MARGIN = 10;
const REENTER_ARM_MARGIN = 20;
const REENTER_ARM_MARGIN_C = 20;
const WORLD_B_MAX_DISTANCE_FACTOR = 1.1;
const WORLD_C_MAX_DISTANCE_FACTOR = 1.1;
const REENTER_ARM_MARGIN_D = 20;
const WORLD_D_MAX_DISTANCE_FACTOR = 1.05;
const WORLD_D_ROTATE_REQUIRED = 1.5;
const WORLD_D_FIST_THRESHOLD = 0.5;
const WORLD_D_STRESS_HOLD_MS = 3000;
const WORLD_D_PARTICLE_COUNT = 600;
const OK_HOLD_MS = 3000;

// Fallback in case A.png is missing
const FALLBACK_IMAGE = "https://images.unsplash.com/photo-1563089145-599997674d42?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80";
const MAX_GROWTH = 0.85; 

// Zoom Thresholds
const ZOOM_IN_THRESHOLD = 0.07; // Fingers close together
const ZOOM_OUT_THRESHOLD = 0.15; // Fingers spread apart
const ZOOM_IN_RANGE = 0.045;
const ZOOM_OUT_RANGE = 0.14;

type ViewState = { position: THREE.Vector3; target: THREE.Vector3 };

export function App() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(DEFAULT_IMAGE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [animating, setAnimating] = useState(false);
  const animatingRef = useRef(false); 
  const [growth, setGrowth] = useState(0);
  const [currentWorld, setCurrentWorld] = useState<'A' | 'B' | 'C' | 'D'>('A');
  const [worldBUnlocked, setWorldBUnlocked] = useState(false);
  const [worldCUnlocked, setWorldCUnlocked] = useState(false);
  const [worldDUnlocked, setWorldDUnlocked] = useState(false);
  const [deepZoneReached, setDeepZoneReached] = useState(false);
  const [deepZoneReachedB, setDeepZoneReachedB] = useState(false);
  const [deepZoneReachedC, setDeepZoneReachedC] = useState(false);
  const [okHoldProgress, setOkHoldProgress] = useState(0);
  const [okHoldProgressC, setOkHoldProgressC] = useState(0);
  const [okHoldProgressD, setOkHoldProgressD] = useState(0);
  const [worldDRotationProgress, setWorldDRotationProgress] = useState(0);
  const [worldDStressProgress, setWorldDStressProgress] = useState(0);
  const [worldDStressActive, setWorldDStressActive] = useState(false);
  const [, setWorldCLoading] = useState(false);
  const currentWorldRef = useRef<'A' | 'B' | 'C' | 'D'>('A');
  const worldBUnlockedRef = useRef(false);
  const worldCUnlockedRef = useRef(false);
  const worldDUnlockedRef = useRef(false);
  const deepZoneActiveRef = useRef(false);
  const deepZoneReachedRef = useRef(false);
  const deepZoneReachedBRef = useRef(false);
  const deepZoneReachedCRef = useRef(false);
  const okHoldStartRef = useRef<number | null>(null);
  const okHoldProgressRef = useRef(0);
  const okHoldStartCRef = useRef<number | null>(null);
  const okHoldProgressCRef = useRef(0);
  const okHoldStartDRef = useRef<number | null>(null);
  const okHoldProgressDRef = useRef(0);
  const worldDLastWristAngleRef = useRef<number | null>(null);
  const worldDRotationAccumRef = useRef(0);
  const worldDRotationProgressRef = useRef(0);
  const worldDStressHoldStartRef = useRef<number | null>(null);
  const worldDStressHoldProgressRef = useRef(0);
  const worldDStressActiveRef = useRef(false);
  const worldDFistStrengthRef = useRef(0);
  const worldAImageRef = useRef<string | null>(DEFAULT_IMAGE);
  const worldBImageRef = useRef<string | null>(WORLD_B_IMAGE);

  // Hand Gesture State
  const [handControlEnabled, setHandControlEnabled] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const previousHandPos = useRef<{x: number, y: number} | null>(null);
  const smoothedHandPos = useRef<{x: number, y: number} | null>(null);
  const handVelocity = useRef<{x: number, y: number}>({ x: 0, y: 0 });
  const rightHandTargetRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const rightHandZoomTargetRef = useRef(0);
  const rightHandZoomVelocityRef = useRef(0);
  const rightHandActiveRef = useRef(false);
  const handFrameTimeRef = useRef<number | null>(null);
  // Refs for Three.js objects
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const clockRef = useRef<THREE.Clock>(new THREE.Clock());
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const worldCModelRef = useRef<THREE.Object3D | null>(null);
  const worldCLoadingRef = useRef<Promise<THREE.Object3D> | null>(null);
  const worldCReadyRef = useRef(false);
  const worldCPendingRef = useRef(false);
  const worldCLightsRef = useRef<THREE.Light[] | null>(null);
  const worldDGroupRef = useRef<THREE.Group | null>(null);
  const worldDTopRef = useRef<THREE.Mesh | null>(null);
  const worldDBottomRef = useRef<THREE.Mesh | null>(null);
  const worldDGrooveRef = useRef<THREE.Mesh | null>(null);
  const worldDParticlesRef = useRef<THREE.Points | null>(null);
  const worldDParticleDataRef = useRef<{ positions: Float32Array; velocities: Float32Array; basePositions: Float32Array } | null>(null);
  const worldDMaterialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const worldDGrooveMaterialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const worldDLightsRef = useRef<THREE.Light[] | null>(null);
  const worldDReadyRef = useRef(false);
  const textureCacheRef = useRef<Record<string, THREE.Texture>>({});
  const textureLoadingRef = useRef<Record<string, Promise<THREE.Texture>>>({});
  const textureLoaderRef = useRef<THREE.TextureLoader | null>(null);
  const defaultViewRef = useRef<ViewState | null>(null);
  const worldAViewRef = useRef<ViewState | null>(null);
  const worldBViewRef = useRef<ViewState | null>(null);
  const worldCViewRef = useRef<ViewState | null>(null);
  const worldDViewRef = useRef<ViewState | null>(null);
  const worldCVisitedRef = useRef(false);
  const worldDVisitedRef = useRef(false);
  const worldBVisitedRef = useRef(false);
  const worldBReturnDistanceRef = useRef<number | null>(null);
  const worldBReturnArmedRef = useRef(false);
  const worldBEntryArmedRef = useRef(false);
  const worldCEntryArmedRef = useRef(false);
  const worldCReturnDistanceRef = useRef<number | null>(null);
  const worldCReturnArmedRef = useRef(false);
  const worldDEntryArmedRef = useRef(false);
  const worldDReturnDistanceRef = useRef<number | null>(null);
  const worldDReturnArmedRef = useRef(false);
  const worldAGrowthRef = useRef(0);
  const worldBGrowthRef = useRef(MAX_GROWTH);
  const defaultMaxDistanceRef = useRef<number | null>(null);
  const initialAGrowthCompletedRef = useRef(false);

  // Parameters
  const paramsRef = useRef({
    displacement: 80,
    noiseSpeed: 0.2,
    pointSize: 3.5,
    threshold: 0.15,
    opacity: 0.9,
    bloomStrength: 0.3,
    growthSpeed: 0.005,
  });

  const getTextureLoader = () => {
    if (!textureLoaderRef.current) {
      const loader = new THREE.TextureLoader();
      loader.crossOrigin = 'anonymous';
      textureLoaderRef.current = loader;
    }
    return textureLoaderRef.current as THREE.TextureLoader;
  };

  const applyTextureOrientation = (texture: THREE.Texture, url: string) => {
    if (url !== WORLD_B_IMAGE) return;
    if (texture.flipY !== false) {
      texture.flipY = false;
      texture.needsUpdate = true;
    }
  };

  const loadTexture = useCallback((url: string) => {
    const cachedTexture = textureCacheRef.current[url];
    if (cachedTexture) {
      applyTextureOrientation(cachedTexture, url);
      return Promise.resolve(cachedTexture);
    }
    if (textureLoadingRef.current[url]) {
      return textureLoadingRef.current[url];
    }

    const loader = getTextureLoader();
    const promise = new Promise<THREE.Texture>((resolve, reject) => {
      loader.load(
        url,
        (texture) => {
          applyTextureOrientation(texture, url);
          textureCacheRef.current[url] = texture;
          delete textureLoadingRef.current[url];
          resolve(texture);
        },
        undefined,
        (err) => {
          delete textureLoadingRef.current[url];
          reject(err);
        }
      );
    });

    textureLoadingRef.current[url] = promise;
    return promise;
  }, []);

  const buildGeometryFromTexture = useCallback((texture: THREE.Texture) => {
    const img = texture.image as { width?: number; height?: number };
    const imgWidth = img?.width || 1;
    const imgHeight = img?.height || 1;

    const maxDim = 600;
    let rw = imgWidth;
    let rh = imgHeight;
    if (imgWidth > maxDim || imgHeight > maxDim) {
      const aspect = imgWidth / imgHeight;
      if (imgWidth > imgHeight) {
        rw = maxDim;
        rh = maxDim / aspect;
      } else {
        rh = maxDim;
        rw = maxDim * aspect;
      }
    }

    return new THREE.PlaneGeometry(rw, rh, Math.floor(rw), Math.floor(rh));
  }, []);

  const applyTextureToScene = useCallback((texture: THREE.Texture, resetGrowth: boolean) => {
    if (!pointsRef.current || !materialRef.current) return;

    const newGeometry = buildGeometryFromTexture(texture);
    pointsRef.current.geometry.dispose();
    pointsRef.current.geometry = newGeometry;
    materialRef.current.uniforms.uTexture.value = texture;
    materialRef.current.uniforms.uTexture.needsUpdate = true;

    if (resetGrowth) {
      materialRef.current.uniforms.uGrowth.value = 0.0;
      animatingRef.current = true;
      setAnimating(true);
      setGrowth(0);
    }
  }, [buildGeometryFromTexture]);

  const preloadTexture = useCallback((url: string | null) => {
    if (!url) return;
    void loadTexture(url).catch((err) => {
      console.warn(`Failed to preload texture: ${url}`, err);
    });
  }, [loadTexture]);

  const swapTexture = useCallback(async (
    textureUrl: string,
    options: { resetGrowth?: boolean; isDefault?: boolean } = {}
  ) => {
    const { resetGrowth = true, isDefault = false } = options;
    try {
      const texture = await loadTexture(textureUrl);
      applyTextureToScene(texture, resetGrowth);
    } catch (err) {
      if (isDefault) {
        console.warn(`Default image at ${textureUrl} failed to load. Falling back.`, err);
        try {
          const fallbackTexture = await loadTexture(FALLBACK_IMAGE);
          if (currentWorldRef.current === 'A') {
            worldAImageRef.current = FALLBACK_IMAGE;
          } else {
            worldBImageRef.current = FALLBACK_IMAGE;
          }
          applyTextureToScene(fallbackTexture, resetGrowth);
        } catch (fallbackErr) {
          console.error("Fallback texture failed to load", fallbackErr);
          setError("Could not load image. Please try a valid image file or URL.");
        }
      } else {
        console.error("Texture Load Error:", err);
        setError("Could not load image. Please try a valid image file or URL.");
      }
    } finally {
      setLoading(false);
    }
  }, [applyTextureToScene, loadTexture]);

  const ensureWorldCLights = useCallback(() => {
    if (!sceneRef.current) return;
    if (worldCLightsRef.current) return;
    const hemi = new THREE.HemisphereLight(0xffffff, 0x080820, 0.9);
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(6, 10, 8);
    hemi.visible = false;
    dir.visible = false;
    sceneRef.current.add(hemi, dir);
    worldCLightsRef.current = [hemi, dir];
  }, []);

  const fitWorldCModel = useCallback((model: THREE.Object3D) => {
    const box = new THREE.Box3().setFromObject(model);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const target = 320;
    const scale = target / maxDim;
    model.scale.setScalar(scale);
    box.setFromObject(model);
    const center = new THREE.Vector3();
    box.getCenter(center);
    model.position.sub(center);
  }, []);

  const buildWorldCParticles = useCallback((model: THREE.Object3D) => {
    const baseColor = new THREE.Color(0x6fdedc);
    const highlight = new THREE.Color(0xcfffff);
    let total = 0;
    model.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const geometry = (child as THREE.Mesh).geometry as THREE.BufferGeometry;
        const position = geometry.getAttribute('position');
        if (position) {
          total += position.count;
        }
      }
    });
    const targetCount = 12000;
    const step = total > 0 ? Math.max(1, Math.floor(total / targetCount)) : 1;
    const positions: number[] = [];
    const colors: number[] = [];
    const temp = new THREE.Vector3();
    model.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        const geometry = mesh.geometry as THREE.BufferGeometry;
        const position = geometry.getAttribute('position');
        if (!position) return;
        for (let i = 0; i < position.count; i += step) {
          temp.fromBufferAttribute(position, i);
          temp.applyMatrix4(mesh.matrixWorld);
          positions.push(temp.x, temp.y, temp.z);
          const t = Math.random();
          const color = baseColor.clone().lerp(highlight, 0.35 + 0.65 * t);
          colors.push(color.r, color.g, color.b);
        }
      }
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();
    const material = new THREE.PointsMaterial({
      size: 2.3,
      color: 0xffffff,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    return new THREE.Points(geometry, material);
  }, []);

  const loadWorldCModel = useCallback(() => {
    if (worldCReadyRef.current && worldCModelRef.current) {
      return Promise.resolve(worldCModelRef.current);
    }
    if (worldCLoadingRef.current) {
      return worldCLoadingRef.current;
    }
    if (!sceneRef.current) {
      return Promise.reject(new Error("Scene not ready"));
    }
    setWorldCLoading(true);
    const loader = new GLTFLoader();
    const promise = new Promise<THREE.Object3D>((resolve, reject) => {
      loader.load(
        WORLD_C_MODEL,
        (gltf) => {
          ensureWorldCLights();
          const model = gltf.scene;
          fitWorldCModel(model);
          model.updateMatrixWorld(true);
          const points = buildWorldCParticles(model);
          points.visible = false;
          sceneRef.current.add(points);
          worldCModelRef.current = points;
          worldCReadyRef.current = true;
          worldCLoadingRef.current = null;
          setWorldCLoading(false);
          resolve(points);
        },
        undefined,
        (err) => {
          worldCLoadingRef.current = null;
          setWorldCLoading(false);
          reject(err);
        }
      );
    });
    worldCLoadingRef.current = promise;
    return promise;
  }, [ensureWorldCLights, fitWorldCModel, buildWorldCParticles]);

  const preloadWorldCModel = useCallback(() => {
    void loadWorldCModel().catch((err) => {
      console.warn("Failed to preload World C model", err);
    });
  }, [loadWorldCModel]);

  const setWorldVisibility = (world: 'A' | 'B' | 'C' | 'D') => {
    if (pointsRef.current) {
      pointsRef.current.visible = world === 'A' || world === 'B';
    }
    if (worldCModelRef.current) {
      worldCModelRef.current.visible = world === 'C';
    }
    if (worldDGroupRef.current) {
      worldDGroupRef.current.visible = world === 'D';
    }
    if (worldCLightsRef.current) {
      worldCLightsRef.current.forEach((light) => {
        light.visible = world === 'C';
      });
    }
    if (worldDLightsRef.current) {
      worldDLightsRef.current.forEach((light) => {
        light.visible = world === 'D';
      });
    }
  };

  const enterWorldC = () => {
    if (worldCReadyRef.current) {
      switchWorld('C');
      return;
    }
    worldCPendingRef.current = true;
    setWorldCLoading(true);
    void loadWorldCModel()
      .then(() => {
        if (!worldCPendingRef.current) return;
        worldCPendingRef.current = false;
        setWorldCLoading(false);
        if (currentWorldRef.current === 'B') {
          switchWorld('C');
        }
      })
      .catch((err) => {
        console.warn("Failed to load World C model", err);
        worldCPendingRef.current = false;
        setWorldCLoading(false);
      });
  };

  const ensureWorldDLights = useCallback(() => {
    if (!sceneRef.current) return;
    if (worldDLightsRef.current) return;
    const ambient = new THREE.AmbientLight(0xffffff, 0.45);
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(8, 12, 6);
    const rim = new THREE.DirectionalLight(0x9fdfff, 0.6);
    rim.position.set(-8, -10, 10);
    ambient.visible = false;
    key.visible = false;
    rim.visible = false;
    sceneRef.current.add(ambient, key, rim);
    worldDLightsRef.current = [ambient, key, rim];
  }, []);

  const buildWorldDParticles = useCallback((group: THREE.Group) => {
    const count = WORLD_D_PARTICLE_COUNT;
    const positions = new Float32Array(count * 3);
    const basePositions = new Float32Array(count * 3);
    const velocities = new Float32Array(count * 3);
    const baseRadius = 230;
    const radiusJitter = 80;
    for (let i = 0; i < count; i++) {
      const offset = i * 3;
      const dir = new THREE.Vector3(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
      dir.normalize();
      const radius = baseRadius + Math.random() * radiusJitter;
      const pos = dir.multiplyScalar(radius);
      positions[offset] = pos.x;
      positions[offset + 1] = pos.y;
      positions[offset + 2] = pos.z;
      basePositions[offset] = pos.x;
      basePositions[offset + 1] = pos.y;
      basePositions[offset + 2] = pos.z;
      velocities[offset] = 0;
      velocities[offset + 1] = 0;
      velocities[offset + 2] = 0;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0xaefcff,
      size: 2.2,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
    });
    const particles = new THREE.Points(geometry, material);
    group.add(particles);
    worldDParticlesRef.current = particles;
    worldDParticleDataRef.current = { positions, velocities, basePositions };
  }, []);

  const ensureWorldDModel = useCallback(() => {
    if (worldDReadyRef.current) return;
    if (!sceneRef.current) return;
    ensureWorldDLights();
    const group = new THREE.Group();
    group.visible = false;
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: 0xdedede,
      roughness: 0.85,
      metalness: 0.05,
    });
    const grooveMaterial = new THREE.MeshStandardMaterial({
      color: 0x8a8a8a,
      roughness: 0.9,
      metalness: 0.02,
      emissive: new THREE.Color(0x111111),
      emissiveIntensity: 0.4,
    });
    const radius = 120;
    const sphereGeometry = new THREE.SphereGeometry(radius, 64, 64);
    const top = new THREE.Mesh(sphereGeometry, bodyMaterial);
    const bottom = new THREE.Mesh(sphereGeometry, bodyMaterial);
    top.position.y = radius * 0.42;
    bottom.position.y = -radius * 0.42;
    top.userData.baseY = top.position.y;
    bottom.userData.baseY = bottom.position.y;
    const grooveGeometry = new THREE.TorusGeometry(radius * 0.92, radius * 0.08, 16, 80);
    const groove = new THREE.Mesh(grooveGeometry, grooveMaterial);
    groove.rotation.x = Math.PI / 2;
    const topSpikeGroup = new THREE.Group();
    const bottomSpikeGroup = new THREE.Group();
    const spikeBaseHeight = 10;
    const spikeGeometry = new THREE.ConeGeometry(3.5, spikeBaseHeight, 8);
    const spikeMaterial = new THREE.MeshStandardMaterial({
      color: 0xf2f2f2,
      roughness: 0.9,
      metalness: 0.05,
    });
    const spikeCount = 180;
    const spikesPerHemisphere = Math.floor(spikeCount / 2);
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    const minHemisphereY = 0.2;
    const normal = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);

    for (let i = 0; i < spikesPerHemisphere; i++) {
      const t = (i + 0.5) / spikesPerHemisphere;
      const y = minHemisphereY + (1 - minHemisphereY) * t;
      const radiusXZ = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = goldenAngle * i;
      normal.set(Math.cos(theta) * radiusXZ, y, Math.sin(theta) * radiusXZ);
      const height = spikeBaseHeight * (0.85 + 0.3 * ((i % 7) / 6));
      const cone = new THREE.Mesh(spikeGeometry, spikeMaterial);
      cone.scale.set(1, height / spikeBaseHeight, 1);
      cone.position.copy(normal).multiplyScalar(radius + height * 0.5);
      cone.quaternion.setFromUnitVectors(up, normal);
      topSpikeGroup.add(cone);
    }

    for (let i = 0; i < spikesPerHemisphere; i++) {
      const t = (i + 0.5) / spikesPerHemisphere;
      const y = -(minHemisphereY + (1 - minHemisphereY) * t);
      const radiusXZ = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = goldenAngle * (i + spikesPerHemisphere * 0.5);
      normal.set(Math.cos(theta) * radiusXZ, y, Math.sin(theta) * radiusXZ);
      const height = spikeBaseHeight * (0.85 + 0.3 * ((i % 7) / 6));
      const cone = new THREE.Mesh(spikeGeometry, spikeMaterial);
      cone.scale.set(1, height / spikeBaseHeight, 1);
      cone.position.copy(normal).multiplyScalar(radius + height * 0.5);
      cone.quaternion.setFromUnitVectors(up, normal);
      bottomSpikeGroup.add(cone);
    }

    top.add(topSpikeGroup);
    bottom.add(bottomSpikeGroup);
    group.add(top, bottom, groove);
    buildWorldDParticles(group);
    sceneRef.current.add(group);
    worldDGroupRef.current = group;
    worldDTopRef.current = top;
    worldDBottomRef.current = bottom;
    worldDGrooveRef.current = groove;
    worldDMaterialRef.current = bodyMaterial;
    worldDGrooveMaterialRef.current = grooveMaterial;
    worldDReadyRef.current = true;
  }, [buildWorldDParticles, ensureWorldDLights]);

  const enterWorldD = () => {
    ensureWorldDModel();
    switchWorld('D');
  };

const cloneViewState = (state: ViewState) => ({
    position: state.position.clone(),
    target: state.target.clone(),
  });

  const captureViewState = () => {
    if (!cameraRef.current || !controlsRef.current) return null;
    return {
      position: cameraRef.current.position.clone(),
      target: controlsRef.current.target.clone(),
    };
  };

  const applyViewState = (state: ViewState | null) => {
    if (!state || !cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.copy(state.position);
    controlsRef.current.target.copy(state.target);
    controlsRef.current.update();
  };

  const syncGrowthRef = (value: number) => {
    if (currentWorldRef.current === 'A') {
      worldAGrowthRef.current = value;
    } else if (currentWorldRef.current === 'B') {
      worldBGrowthRef.current = value;
    }
  };

  const setGrowthInstant = (value: number) => {
    animatingRef.current = false;
    setAnimating(false);
    setGrowth(value);
    syncGrowthRef(value);
    if (materialRef.current) {
      materialRef.current.uniforms.uGrowth.value = value;
    }
    if (currentWorldRef.current === 'A' && value >= MAX_GROWTH) {
      initialAGrowthCompletedRef.current = true;
    }
  };

  const getWorldMaxZoomDistance = (world: 'A' | 'B' | 'C' | 'D') => {
    if (world === 'B') {
      return (defaultMaxDistanceRef.current ?? MAX_ZOOM_DISTANCE) * WORLD_B_MAX_DISTANCE_FACTOR;
    }
    if (world === 'C') {
      return (defaultMaxDistanceRef.current ?? MAX_ZOOM_DISTANCE) * WORLD_C_MAX_DISTANCE_FACTOR;
    }
    if (world === 'D') {
      return (defaultMaxDistanceRef.current ?? MAX_ZOOM_DISTANCE) * WORLD_D_MAX_DISTANCE_FACTOR;
    }
    return MAX_ZOOM_DISTANCE;
  };

  const applyZoomLimits = (world: 'A' | 'B' | 'C' | 'D') => {
    if (!controlsRef.current) return;
    controlsRef.current.minDistance = MIN_ZOOM_DISTANCE;
    controlsRef.current.maxDistance = getWorldMaxZoomDistance(world);
  };

  const updateWorldBReturnDistance = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    const distToTarget = cameraRef.current.position.distanceTo(controlsRef.current.target);
    const maxDistance = getWorldMaxZoomDistance('B');
    worldBReturnDistanceRef.current = Math.min(distToTarget + RETURN_MARGIN, maxDistance);
  };

  const updateWorldCReturnDistance = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    const distToTarget = cameraRef.current.position.distanceTo(controlsRef.current.target);
    const maxDistance = getWorldMaxZoomDistance('C');
    worldCReturnDistanceRef.current = Math.min(distToTarget + RETURN_MARGIN, maxDistance);
  };

  const updateWorldDReturnDistance = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    const distToTarget = cameraRef.current.position.distanceTo(controlsRef.current.target);
    const maxDistance = getWorldMaxZoomDistance('D');
    worldDReturnDistanceRef.current = Math.min(distToTarget + RETURN_MARGIN, maxDistance);
  };

  const snapshotCurrentWorld = () => {
    const viewState = captureViewState();
    if (viewState) {
      if (currentWorldRef.current === 'A') {
        worldAViewRef.current = viewState;
      } else if (currentWorldRef.current === 'B') {
        worldBViewRef.current = viewState;
      } else if (currentWorldRef.current === 'C') {
        worldCViewRef.current = viewState;
      } else {
        worldDViewRef.current = viewState;
      }
    }
    syncGrowthRef(growth);
  };

  const setOkProgress = (progress: number) => {
    okHoldProgressRef.current = progress;
    setOkHoldProgress(progress);
  };

  const resetOkHold = () => {
    okHoldStartRef.current = null;
    if (okHoldProgressRef.current !== 0) {
        setOkProgress(0);
    }
  };

  const setOkProgressC = (progress: number) => {
    okHoldProgressCRef.current = progress;
    setOkHoldProgressC(progress);
  };

  const setOkProgressD = (progress: number) => {
    okHoldProgressDRef.current = progress;
    setOkHoldProgressD(progress);
  };

  const resetOkHoldC = () => {
    okHoldStartCRef.current = null;
    if (okHoldProgressCRef.current !== 0) {
        setOkProgressC(0);
    }
  };

  const resetOkHoldD = () => {
    okHoldStartDRef.current = null;
    if (okHoldProgressDRef.current !== 0) {
        setOkProgressD(0);
    }
  };

  const unlockWorldB = () => {
    if (worldBUnlockedRef.current) return;
    worldBUnlockedRef.current = true;
    setWorldBUnlocked(true);
  };

  const unlockWorldC = () => {
    if (worldCUnlockedRef.current) return;
    worldCUnlockedRef.current = true;
    setWorldCUnlocked(true);
  };

  const unlockWorldD = () => {
    if (worldDUnlockedRef.current) return;
    worldDUnlockedRef.current = true;
    setWorldDUnlocked(true);
  };

  const switchWorld = (nextWorld: 'A' | 'B' | 'C' | 'D') => {
    if (currentWorldRef.current === nextWorld) return;
    const previousWorld = currentWorldRef.current;

    if (nextWorld === 'C' && !worldCReadyRef.current) {
        enterWorldC();
        return;
    }
    if (nextWorld === 'D' && !worldDReadyRef.current) {
        ensureWorldDModel();
    }

    snapshotCurrentWorld();

    if (currentWorldRef.current === 'A' && nextWorld === 'B' && !initialAGrowthCompletedRef.current) {
        worldAGrowthRef.current = MAX_GROWTH;
        initialAGrowthCompletedRef.current = true;
    }

    currentWorldRef.current = nextWorld;
    setCurrentWorld(nextWorld);
    if (previousWorld === 'D' && nextWorld !== 'D') {
        worldDStressActiveRef.current = false;
        worldDStressHoldStartRef.current = null;
        if (worldDStressHoldProgressRef.current < 1) {
            worldDStressHoldProgressRef.current = 0;
            setWorldDStressProgress(0);
        }
        setWorldDStressActive(false);
        worldDFistStrengthRef.current = 0;
        worldDLastWristAngleRef.current = null;
    }
    if (nextWorld === 'A' || nextWorld === 'B') {
        const nextImage = nextWorld === 'A' ? worldAImageRef.current : worldBImageRef.current;
        if (nextImage) {
            const isDefaultTexture = nextImage === DEFAULT_IMAGE || nextImage === WORLD_B_IMAGE;
            void swapTexture(nextImage, { resetGrowth: false, isDefault: isDefaultTexture });
            const preloadTarget = nextWorld === 'A' ? worldBImageRef.current : worldAImageRef.current;
            preloadTexture(preloadTarget);
            preloadWorldCModel();
            ensureWorldDModel();
        }
    }

    setWorldVisibility(nextWorld);

    if (nextWorld === 'B') {
        if (!worldBVisitedRef.current) {
            worldBVisitedRef.current = true;
            if (defaultViewRef.current) {
                const initialView = cloneViewState(defaultViewRef.current);
                const maxDistance = getWorldMaxZoomDistance('B');
                const offset = initialView.position.clone().sub(initialView.target);
                const currentDistance = offset.length();
                if (currentDistance > 0) {
                    offset.setLength(maxDistance);
                    initialView.position.copy(initialView.target).add(offset);
                }
                worldBViewRef.current = initialView;
                applyViewState(initialView);
            }
        } else {
            if (worldBViewRef.current) {
                applyViewState(worldBViewRef.current);
            }
        }
        setGrowthInstant(MAX_GROWTH);
        updateWorldBReturnDistance();
        worldBReturnArmedRef.current = false;
        worldCEntryArmedRef.current = false;
    } else if (nextWorld === 'C') {
        if (!worldCVisitedRef.current) {
            worldCVisitedRef.current = true;
            if (defaultViewRef.current) {
                const initialView = cloneViewState(defaultViewRef.current);
                const maxDistance = getWorldMaxZoomDistance('C');
                const offset = initialView.position.clone().sub(initialView.target);
                const currentDistance = offset.length();
                if (currentDistance > 0) {
                    offset.setLength(maxDistance);
                    initialView.position.copy(initialView.target).add(offset);
                }
                worldCViewRef.current = initialView;
                applyViewState(initialView);
            }
        } else if (worldCViewRef.current) {
            applyViewState(worldCViewRef.current);
        }
        updateWorldCReturnDistance();
        worldCReturnArmedRef.current = false;
        worldDEntryArmedRef.current = false;
    } else if (nextWorld === 'D') {
        if (!worldDVisitedRef.current) {
            worldDVisitedRef.current = true;
            worldDRotationAccumRef.current = 0;
            worldDRotationProgressRef.current = 0;
            setWorldDRotationProgress(0);
            worldDStressHoldStartRef.current = null;
            worldDStressHoldProgressRef.current = 0;
            setWorldDStressProgress(0);
            worldDStressActiveRef.current = false;
            setWorldDStressActive(false);
            worldDLastWristAngleRef.current = null;
            if (defaultViewRef.current) {
                const initialView = cloneViewState(defaultViewRef.current);
                const maxDistance = getWorldMaxZoomDistance('D');
                const offset = initialView.position.clone().sub(initialView.target);
                const currentDistance = offset.length();
                if (currentDistance > 0) {
                    offset.setLength(maxDistance);
                    initialView.position.copy(initialView.target).add(offset);
                }
                worldDViewRef.current = initialView;
                applyViewState(initialView);
            }
        } else if (worldDViewRef.current) {
            applyViewState(worldDViewRef.current);
        }
        updateWorldDReturnDistance();
        worldDReturnArmedRef.current = false;
    } else {
        if (worldAViewRef.current) {
            applyViewState(worldAViewRef.current);
        }
        if (initialAGrowthCompletedRef.current) {
            worldAGrowthRef.current = MAX_GROWTH;
            setGrowthInstant(MAX_GROWTH);
        } else {
            setGrowthInstant(worldAGrowthRef.current);
        }
        worldBEntryArmedRef.current = false;
    }

    applyZoomLimits(nextWorld);
    resetOkHold();
    resetOkHoldC();
    resetOkHoldD();
    deepZoneActiveRef.current = false;
  };

  const setWorldImage = (nextImage: string) => {
    if (currentWorldRef.current === 'A') {
        worldAImageRef.current = nextImage;
    } else {
        worldBImageRef.current = nextImage;
    }
    setImageSrc(nextImage);
  };

  const handleDemoImage = () => {
    const demoImage = currentWorldRef.current === 'A' ? DEFAULT_IMAGE : WORLD_B_IMAGE;
    setWorldImage(demoImage);
  };

  const isOkGesture = (hand: any) => {
    if (!hand || hand.length < 21) return false;
    const dist = (a: any, b: any) => Math.sqrt(
        Math.pow(a.x - b.x, 2) +
        Math.pow(a.y - b.y, 2) +
        Math.pow(a.z - b.z, 2)
    );
    const palmSize = dist(hand[0], hand[9]) || 1;
    const pinch = dist(hand[4], hand[8]) < palmSize * 0.35;
    const middleExtended = hand[12].y < hand[10].y;
    const ringExtended = hand[16].y < hand[14].y;
    const pinkyExtended = hand[20].y < hand[18].y;
    return pinch && middleExtended && ringExtended && pinkyExtended;
  };

  const getWristRotationAngle = (hand: any) => {
    if (!hand || hand.length < 10) return null;
    const wrist = hand[0];
    const mid = hand[9];
    const dx = mid.x - wrist.x;
    const dy = mid.y - wrist.y;
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) return null;
    return Math.atan2(dy, dx);
  };

  const getFistStrength = (hand: any) => {
    if (!hand || hand.length < 21) return 0;
    const dist = (a: any, b: any) => Math.sqrt(
        Math.pow(a.x - b.x, 2) +
        Math.pow(a.y - b.y, 2) +
        Math.pow(a.z - b.z, 2)
    );
    const palmCenter = {
        x: (hand[0].x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5,
        y: (hand[0].y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5,
        z: (hand[0].z + hand[5].z + hand[9].z + hand[13].z + hand[17].z) / 5,
    };
    const palmSize = dist(hand[5], hand[17]) || dist(hand[0], hand[9]) || 1;
    const fingers = [
        [8, 6, 5],
        [12, 10, 9],
        [16, 14, 13],
        [20, 18, 17],
    ];
    let sum = 0;
    fingers.forEach(([tipIdx, pipIdx, mcpIdx]) => {
        const tip = hand[tipIdx];
        const pip = hand[pipIdx];
        const mcp = hand[mcpIdx];
        const tipDist = dist(tip, palmCenter);
        const mcpDist = dist(mcp, palmCenter);
        const tipScore = THREE.MathUtils.clamp((palmSize * 0.95 - tipDist) / (palmSize * 0.5), 0, 1);
        const ratio = tipDist / (mcpDist || 1);
        const ratioScore = THREE.MathUtils.clamp((1.1 - ratio) / 0.6, 0, 1);

        const v1x = mcp.x - pip.x;
        const v1y = mcp.y - pip.y;
        const v1z = mcp.z - pip.z;
        const v2x = tip.x - pip.x;
        const v2y = tip.y - pip.y;
        const v2z = tip.z - pip.z;
        const v1Len = Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
        const v2Len = Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
        let angleScore = 0;
        if (v1Len > 1e-5 && v2Len > 1e-5) {
            const cos = THREE.MathUtils.clamp((v1x * v2x + v1y * v2y + v1z * v2z) / (v1Len * v2Len), -1, 1);
            const angle = Math.acos(cos);
            angleScore = THREE.MathUtils.clamp((Math.PI - angle) / (Math.PI * 0.7), 0, 1);
        }

        const fingerScore = tipScore * 0.5 + ratioScore * 0.3 + angleScore * 0.2;
        sum += fingerScore;
    });
    const thumbTip = hand[4];
    const indexMcp = hand[5];
    const thumbToPalm = dist(thumbTip, palmCenter);
    const thumbToIndex = dist(thumbTip, indexMcp);
    const thumbScore = THREE.MathUtils.clamp(
        (palmSize * 0.65 - Math.min(thumbToPalm, thumbToIndex)) / (palmSize * 0.65),
        0,
        1
    );
    const combined = (sum / fingers.length) * 0.85 + thumbScore * 0.15;
    return Math.max(0, Math.min(1, combined));
  };
    
  const updateOkHold = (isOk: boolean) => {
    if (currentWorldRef.current !== 'A' || worldBUnlockedRef.current || !deepZoneActiveRef.current || !isOk) {
        resetOkHold();
        return;
    }

    const now = performance.now();
    if (!okHoldStartRef.current) {
        okHoldStartRef.current = now;
    }
    const elapsed = now - okHoldStartRef.current;
    const progress = Math.min(elapsed / OK_HOLD_MS, 1);
    if (progress !== okHoldProgressRef.current) {
        setOkProgress(progress);
    }
    if (progress >= 1) {
        unlockWorldB();
        switchWorld('B');
    }
  };

  const updateOkHoldC = (isOk: boolean) => {
    if (currentWorldRef.current !== 'B' || worldCUnlockedRef.current || !deepZoneActiveRef.current || !isOk) {
        resetOkHoldC();
        return;
    }

    const now = performance.now();
    if (!okHoldStartCRef.current) {
        okHoldStartCRef.current = now;
    }
    const elapsed = now - okHoldStartCRef.current;
    const progress = Math.min(elapsed / OK_HOLD_MS, 1);
    if (progress !== okHoldProgressCRef.current) {
        setOkProgressC(progress);
    }
    if (progress >= 1) {
        unlockWorldC();
        enterWorldC();
    }
  };

  const updateOkHoldD = (isOk: boolean) => {
    if (currentWorldRef.current !== 'C' || worldDUnlockedRef.current || !deepZoneActiveRef.current || !isOk) {
        resetOkHoldD();
        return;
    }

    const now = performance.now();
    if (!okHoldStartDRef.current) {
        okHoldStartDRef.current = now;
    }
    const elapsed = now - okHoldStartDRef.current;
    const progress = Math.min(elapsed / OK_HOLD_MS, 1);
    if (progress !== okHoldProgressDRef.current) {
        setOkProgressD(progress);
    }
    if (progress >= 1) {
        unlockWorldD();
        enterWorldD();
    }
  };

  // --- Hand Tracking Logic ---

  useEffect(() => {
    if (!handControlEnabled || !videoRef.current) return;

    let handsInstance: any = null;
    let cameraInstance: any = null;
    let isActive = true;

    const onResults = (results: Results) => {
        if (!isActive || !controlsRef.current || !cameraRef.current) return;

        // 1. Resolve Right/Left Hands
        let rightHandIndex = -1;
        let leftHandIndex = -1;
        if (results.multiHandedness) {
            for (let i = 0; i < results.multiHandedness.length; i++) {
                // In selfieMode=true, "Right" label corresponds to the user's actual Right hand
                if (results.multiHandedness[i].label === 'Right') {
                    rightHandIndex = i;
                }
                if (results.multiHandedness[i].label === 'Left') {
                    leftHandIndex = i;
                }
            }
        }

        const landmarks = results.multiHandLandmarks;

        if (rightHandIndex !== -1 && landmarks && landmarks[rightHandIndex]) {
            const hand = landmarks[rightHandIndex];
            rightHandActiveRef.current = true;

            // --- Panning Logic (Wrist) ---
            const rawX = hand[0].x;
            const rawY = hand[0].y;

            if (!smoothedHandPos.current) {
                smoothedHandPos.current = { x: rawX, y: rawY };
            } else {
                // Low-pass filter to reduce jitter.
                smoothedHandPos.current = {
                    x: THREE.MathUtils.lerp(smoothedHandPos.current.x, rawX, 0.2),
                    y: THREE.MathUtils.lerp(smoothedHandPos.current.y, rawY, 0.2),
                };
            }

            if (previousHandPos.current) {
                // Sensitivity factor
                const sensitivity = 140;

                // Calculate deltas
                // Note: In selfie mode, x moves normal (left is 0, right is 1)
                const deltaX = (smoothedHandPos.current.x - previousHandPos.current.x) * sensitivity;
                const deltaY = (smoothedHandPos.current.y - previousHandPos.current.y) * sensitivity;
                const deadZone = 0.18;
                const targetX = Math.abs(deltaX) < deadZone ? 0 : deltaX;
                const targetY = Math.abs(deltaY) < deadZone ? 0 : deltaY;
                rightHandTargetRef.current.x = THREE.MathUtils.lerp(rightHandTargetRef.current.x, targetX, 0.35);
                rightHandTargetRef.current.y = THREE.MathUtils.lerp(rightHandTargetRef.current.y, targetY, 0.35);
            } else {
                rightHandTargetRef.current.x = 0;
                rightHandTargetRef.current.y = 0;
            }

            previousHandPos.current = { x: smoothedHandPos.current.x, y: smoothedHandPos.current.y };

            // --- Continuous Zoom Logic (Thumb Tip 4 to Index Tip 8) ---
            const thumb = hand[4];
            const index = hand[8];
            const dist = Math.sqrt(
                Math.pow(thumb.x - index.x, 2) +
                Math.pow(thumb.y - index.y, 2)
            );

            let zoomIntent = 0;
            if (dist < ZOOM_IN_THRESHOLD) {
                zoomIntent = THREE.MathUtils.clamp((ZOOM_IN_THRESHOLD - dist) / ZOOM_IN_RANGE, 0, 1);
            } else if (dist > ZOOM_OUT_THRESHOLD) {
                zoomIntent = -THREE.MathUtils.clamp((dist - ZOOM_OUT_THRESHOLD) / ZOOM_OUT_RANGE, 0, 1);
            }
            rightHandZoomTargetRef.current = zoomIntent;
        } else {
            // Reset if hand lost or wrong hand
            previousHandPos.current = null;
            smoothedHandPos.current = null;
            handVelocity.current = { x: 0, y: 0 };
            rightHandZoomVelocityRef.current = 0;
            rightHandTargetRef.current.x = 0;
            rightHandTargetRef.current.y = 0;
            rightHandZoomTargetRef.current = 0;
            rightHandActiveRef.current = false;
        }

        const leftHand = leftHandIndex !== -1 && landmarks && landmarks[leftHandIndex]
            ? landmarks[leftHandIndex]
            : null;
        const okDetected = leftHand ? isOkGesture(leftHand) : false;

        updateOkHold(okDetected);
        updateOkHoldC(okDetected);
        updateOkHoldD(okDetected);

        if (currentWorldRef.current === 'D') {
            if (leftHand) {
                const fistStrength = getFistStrength(leftHand);
                worldDFistStrengthRef.current = THREE.MathUtils.lerp(worldDFistStrengthRef.current, fistStrength, 0.35);
                const activateThreshold = WORLD_D_FIST_THRESHOLD;
                const releaseThreshold = WORLD_D_FIST_THRESHOLD * 0.7;
                const nextStress = worldDStressActiveRef.current
                    ? worldDFistStrengthRef.current >= releaseThreshold
                    : worldDFistStrengthRef.current >= activateThreshold;
                if (nextStress !== worldDStressActiveRef.current) {
                    worldDStressActiveRef.current = nextStress;
                    setWorldDStressActive(nextStress);
                    if (!nextStress && worldDStressHoldProgressRef.current < 1) {
                        worldDStressHoldStartRef.current = null;
                        worldDStressHoldProgressRef.current = 0;
                        setWorldDStressProgress(0);
                    }
                }

                const rotationBlocked = worldDStressActiveRef.current || worldDFistStrengthRef.current >= 0.35;
                if (!rotationBlocked) {
                    const wristAngle = getWristRotationAngle(leftHand);
                    if (wristAngle !== null) {
                        if (worldDLastWristAngleRef.current !== null && worldDGroupRef.current) {
                            const delta = Math.atan2(
                                Math.sin(wristAngle - worldDLastWristAngleRef.current),
                                Math.cos(wristAngle - worldDLastWristAngleRef.current)
                            );
                            worldDGroupRef.current.rotation.y += delta;
                            worldDRotationAccumRef.current = Math.min(worldDRotationAccumRef.current + Math.abs(delta), WORLD_D_ROTATE_REQUIRED);
                            const progress = Math.min(worldDRotationAccumRef.current / WORLD_D_ROTATE_REQUIRED, 1);
                            if (progress >= 1 && worldDRotationProgressRef.current < 1) {
                                worldDRotationProgressRef.current = 1;
                                setWorldDRotationProgress(1);
                            } else if (Math.abs(progress - worldDRotationProgressRef.current) > 0.005) {
                                worldDRotationProgressRef.current = progress;
                                setWorldDRotationProgress(progress);
                            }
                        }
                        worldDLastWristAngleRef.current = wristAngle;
                    } else {
                        worldDLastWristAngleRef.current = null;
                    }
                } else {
                    worldDLastWristAngleRef.current = null;
                }
            } else {
                worldDLastWristAngleRef.current = null;
                worldDFistStrengthRef.current = 0;
                if (worldDStressActiveRef.current) {
                    worldDStressActiveRef.current = false;
                    setWorldDStressActive(false);
                    if (worldDStressHoldProgressRef.current < 1) {
                        worldDStressHoldStartRef.current = null;
                        worldDStressHoldProgressRef.current = 0;
                        setWorldDStressProgress(0);
                    }
                }
            }
        } else {
            worldDLastWristAngleRef.current = null;
        }
    };

    const initMediaPipe = async () => {
        try {
            if (!Hands) {
                throw new Error("Hands class not loaded properly");
            }

            handsInstance = new Hands({
                locateFile: (file: string) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            });

            handsInstance.setOptions({
                maxNumHands: 2, // Allow detection of 2 so we can filter for the Right one even if Left is visible
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5,
                selfieMode: true // IMPORTANT: Mirrors inputs for natural interaction
            });

            handsInstance.onResults(onResults);

            if (videoRef.current) {
                if (!MPCamera) {
                    throw new Error("Camera Utils class not loaded properly");
                }
                cameraInstance = new MPCamera(videoRef.current, {
                    onFrame: async () => {
                        if (!isActive || !videoRef.current || !handsInstance) return;
                        await handsInstance.send({ image: videoRef.current });
                    },
                    width: 320,
                    height: 240
                });
                await cameraInstance.start();
            }
        } catch (e) {
            console.error("Failed to initialize MediaPipe", e);
            setError("Camera access denied or MediaPipe error.");
        }
    };

    initMediaPipe();

    return () => {
      isActive = false;
      rightHandActiveRef.current = false;
      rightHandTargetRef.current.x = 0;
      rightHandTargetRef.current.y = 0;
      rightHandZoomTargetRef.current = 0;
      rightHandZoomVelocityRef.current = 0;
      handVelocity.current = { x: 0, y: 0 };
      previousHandPos.current = null;
      smoothedHandPos.current = null;
      handFrameTimeRef.current = null;
      if (cameraInstance) {
          cameraInstance.stop();
          cameraInstance = null;
      }
      if (handsInstance) {
          const instance = handsInstance;
          handsInstance = null;
          instance.close();
      }
    };
  }, [handControlEnabled]);

  // --- End Hand Tracking Logic ---

  const cleanupScene = () => {
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (rendererRef.current) {
        // Check if the renderer's DOM element is actually a child before removing
        if (mountRef.current && rendererRef.current.domElement.parentNode === mountRef.current) {
            mountRef.current.removeChild(rendererRef.current.domElement);
        }
        rendererRef.current.dispose();
    }
    rendererRef.current = null;
    sceneRef.current = null;
    materialRef.current = null;
    composerRef.current = null;
    pointsRef.current = null;
  };

  const initScene = useCallback((textureUrl: string) => {
    if (!mountRef.current) return;

    if (rendererRef.current) {
        if (mountRef.current.contains(rendererRef.current.domElement)) {
            mountRef.current.removeChild(rendererRef.current.domElement);
        }
        rendererRef.current.dispose();
        rendererRef.current = null;
    }

    const width = window.innerWidth;
    const height = window.innerHeight;

    // Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 1, 5000);
    camera.position.set(0, -200, 350);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;
    controls.enablePan = true;
    controls.enableZoom = true;
    controlsRef.current = controls;

    const initialView: ViewState = {
        position: camera.position.clone(),
        target: controls.target.clone(),
    };
    defaultViewRef.current = initialView;
    worldAViewRef.current = cloneViewState(initialView);
    worldBVisitedRef.current = false;
    worldBViewRef.current = null;
    worldBReturnDistanceRef.current = null;
    worldBReturnArmedRef.current = false;
    worldBEntryArmedRef.current = false;
    worldCVisitedRef.current = false;
    worldCViewRef.current = null;
    worldCEntryArmedRef.current = false;
    worldCReturnDistanceRef.current = null;
    worldCReturnArmedRef.current = false;
    worldCReadyRef.current = false;
    worldCModelRef.current = null;
    worldCLoadingRef.current = null;
    worldCPendingRef.current = false;
    worldCLightsRef.current = null;
    worldDVisitedRef.current = false;
    worldDViewRef.current = null;
    worldDEntryArmedRef.current = false;
    worldDReturnDistanceRef.current = null;
    worldDReturnArmedRef.current = false;
    worldDReadyRef.current = false;
    worldDGroupRef.current = null;
    worldDParticlesRef.current = null;
    worldDParticleDataRef.current = null;
    worldDTopRef.current = null;
    worldDBottomRef.current = null;
    worldDGrooveRef.current = null;
    worldDMaterialRef.current = null;
    worldDGrooveMaterialRef.current = null;
    worldDLightsRef.current = null;
    worldDUnlockedRef.current = false;
    setWorldDUnlocked(false);
    deepZoneReachedCRef.current = false;
    setDeepZoneReachedC(false);
    okHoldStartDRef.current = null;
    okHoldProgressDRef.current = 0;
    setOkHoldProgressD(0);
    worldDRotationAccumRef.current = 0;
    worldDRotationProgressRef.current = 0;
    setWorldDRotationProgress(0);
    worldDStressHoldStartRef.current = null;
    worldDStressHoldProgressRef.current = 0;
    setWorldDStressProgress(0);
    worldDStressActiveRef.current = false;
    setWorldDStressActive(false);
    worldDLastWristAngleRef.current = null;
    worldBUnlockedRef.current = false;
    setWorldBUnlocked(false);
    worldCUnlockedRef.current = false;
    setWorldCUnlocked(false);
    setWorldCLoading(false);
    deepZoneActiveRef.current = false;
    deepZoneReachedRef.current = false;
    setDeepZoneReached(false);
    deepZoneReachedBRef.current = false;
    setDeepZoneReachedB(false);
    okHoldStartRef.current = null;
    okHoldProgressRef.current = 0;
    setOkHoldProgress(0);
    okHoldStartCRef.current = null;
    okHoldProgressCRef.current = 0;
    setOkHoldProgressC(0);
    defaultMaxDistanceRef.current = camera.position.distanceTo(controls.target);
    controls.minDistance = MIN_ZOOM_DISTANCE;
    controls.maxDistance = MAX_ZOOM_DISTANCE;
    initialAGrowthCompletedRef.current = false;

    // Texture Loading
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = 'anonymous';
    
    loader.load(
        textureUrl, 
        (texture) => {
            // Success Callback
            applyTextureOrientation(texture, textureUrl);
            textureCacheRef.current[textureUrl] = texture;
            const geometry = buildGeometryFromTexture(texture);

            const material = new THREE.ShaderMaterial({
                uniforms: {
                    uTime: { value: 0 },
                    uTexture: { value: texture },
                    uDisplacementStrength: { value: paramsRef.current.displacement },
                    uNoiseSpeed: { value: paramsRef.current.noiseSpeed },
                    uPointSize: { value: paramsRef.current.pointSize },
                    uThreshold: { value: paramsRef.current.threshold },
                    uOpacity: { value: paramsRef.current.opacity },
                    uGrowth: { value: 0.0 }
                },
                vertexShader: vertexShader,
                fragmentShader: fragmentShader,
                transparent: true,
                depthWrite: false,
                blending: THREE.NormalBlending,
                side: THREE.DoubleSide
            });
            materialRef.current = material;

            const mesh = new THREE.Points(geometry, material);
            mesh.rotation.x = -Math.PI / 1.6;
            scene.add(mesh);
            pointsRef.current = mesh;

            // Post Processing
            const renderScene = new RenderPass(scene, camera);
            const bloomPass = new UnrealBloomPass(
                new THREE.Vector2(width, height),
                paramsRef.current.bloomStrength,
                0.4,
                0.2
            );
            
            const composer = new EffectComposer(renderer);
            composer.addPass(renderScene);
            composer.addPass(bloomPass);
            composerRef.current = composer;

            const handRight = new THREE.Vector3();
            const handUp = new THREE.Vector3();
            const handPan = new THREE.Vector3();
            const handView = new THREE.Vector3();

            // Animation Loop
            const animate = () => {
                animationFrameRef.current = requestAnimationFrame(animate);
                
                const elapsedTime = clockRef.current.getElapsedTime();
                
                // Only auto-rotate if hands aren't controlling it to avoid fighting
                // Also update damping
                controls.update();

                const lastHandTime = handFrameTimeRef.current;
                const handDelta = lastHandTime === null ? 1 / 60 : Math.min(elapsedTime - lastHandTime, 0.05);
                handFrameTimeRef.current = elapsedTime;

                if (rightHandActiveRef.current) {
                    handVelocity.current.x = THREE.MathUtils.lerp(handVelocity.current.x, rightHandTargetRef.current.x, 0.12);
                    handVelocity.current.y = THREE.MathUtils.lerp(handVelocity.current.y, rightHandTargetRef.current.y, 0.12);
                    rightHandZoomVelocityRef.current = THREE.MathUtils.lerp(rightHandZoomVelocityRef.current, rightHandZoomTargetRef.current, 0.12);

                    const speed = Math.min(handDelta * 60, 2);
                    handRight.set(1, 0, 0).applyQuaternion(camera.quaternion);
                    handUp.set(0, 1, 0).applyQuaternion(camera.quaternion);
                    handPan.set(0, 0, 0)
                        .addScaledVector(handRight, handVelocity.current.x * speed)
                        .addScaledVector(handUp, -handVelocity.current.y * speed);

                    camera.position.add(handPan);
                    controls.target.add(handPan);

                    const handDistToTarget = camera.position.distanceTo(controls.target);
                    const handMaxZoomDistance = getWorldMaxZoomDistance(currentWorldRef.current);
                    const zoomSpeedIn = 36;
                    const zoomSpeedOut = 24;
                    const zoomDelta = rightHandZoomVelocityRef.current * speed;
                    const zoomStep = zoomDelta > 0 ? zoomDelta * zoomSpeedIn : zoomDelta * zoomSpeedOut;

                    if (zoomStep !== 0) {
                        camera.getWorldDirection(handView);
                        if (zoomStep > 0 && handDistToTarget > MIN_ZOOM_DISTANCE) {
                            camera.position.addScaledVector(handView, zoomStep);
                        } else if (zoomStep < 0 && handDistToTarget < handMaxZoomDistance) {
                            camera.position.addScaledVector(handView, zoomStep);
                        }
                    }
                } else {
                    handVelocity.current.x = THREE.MathUtils.lerp(handVelocity.current.x, 0, 0.08);
                    handVelocity.current.y = THREE.MathUtils.lerp(handVelocity.current.y, 0, 0.08);
                    rightHandZoomVelocityRef.current = THREE.MathUtils.lerp(rightHandZoomVelocityRef.current, 0, 0.08);
                }

                                                                const distToTarget = camera.position.distanceTo(controls.target);
                                const isDeepZone = distToTarget <= DEEP_ZONE_DISTANCE;
                                const isWorldA = currentWorldRef.current === 'A';
                                const isWorldB = currentWorldRef.current === 'B';
                                const isWorldC = currentWorldRef.current === 'C';
                                if (isWorldA || isWorldB || isWorldC) {
                                    if (isDeepZone !== deepZoneActiveRef.current) {
                                        deepZoneActiveRef.current = isDeepZone;
                                        if (isDeepZone) {
                                            if (isWorldA && !deepZoneReachedRef.current) {
                                                deepZoneReachedRef.current = true;
                                                setDeepZoneReached(true);
                                            }
                                            if (isWorldB && !deepZoneReachedBRef.current) {
                                                deepZoneReachedBRef.current = true;
                                                setDeepZoneReachedB(true);
                                            }
                                            if (isWorldB && !worldCUnlockedRef.current) {
                                                unlockWorldC();
                                                enterWorldC();
                                            }
                                            if (isWorldC && !deepZoneReachedCRef.current) {
                                                deepZoneReachedCRef.current = true;
                                                setDeepZoneReachedC(true);
                                            }
                                        } else {
                                            if (isWorldA) {
                                                resetOkHold();
                                            }
                                            if (isWorldB) {
                                                resetOkHoldC();
                                            }
                                            if (isWorldC) {
                                                resetOkHoldD();
                                            }
                                        }
                                    }
                                } else if (deepZoneActiveRef.current) {
                                    deepZoneActiveRef.current = false;
                                }

                                const returnDistance = worldBReturnDistanceRef.current ?? RETURN_ZONE_DISTANCE;
                                if (currentWorldRef.current === 'B') {
                                    const armDistance = Math.max(returnDistance - RETURN_ARM_MARGIN, MIN_ZOOM_DISTANCE);
                                    if (!worldBReturnArmedRef.current && distToTarget < armDistance) {
                                        worldBReturnArmedRef.current = true;
                                    }
                                    if (worldBReturnArmedRef.current && distToTarget >= returnDistance) {
                                        switchWorld('A');
                                    }
                                }

                                const returnDistanceC = worldCReturnDistanceRef.current ?? RETURN_ZONE_DISTANCE;
                                if (currentWorldRef.current === 'C') {
                                    const armDistanceC = Math.max(returnDistanceC - RETURN_ARM_MARGIN, MIN_ZOOM_DISTANCE);
                                    if (!worldCReturnArmedRef.current && distToTarget < armDistanceC) {
                                        worldCReturnArmedRef.current = true;
                                    }
                                    if (worldCReturnArmedRef.current && distToTarget >= returnDistanceC) {
                                        switchWorld('B');
                                    }
                                }

                                const returnDistanceD = worldDReturnDistanceRef.current ?? RETURN_ZONE_DISTANCE;
                                if (currentWorldRef.current === 'D') {
                                    const armDistanceD = Math.max(returnDistanceD - RETURN_ARM_MARGIN, MIN_ZOOM_DISTANCE);
                                    if (!worldDReturnArmedRef.current && distToTarget < armDistanceD) {
                                        worldDReturnArmedRef.current = true;
                                    }
                                    if (worldDReturnArmedRef.current && distToTarget >= returnDistanceD) {
                                        switchWorld('C');
                                    }
                                }

                                if (currentWorldRef.current === 'A' && worldBUnlockedRef.current) {
                                    const reenterDistance = DEEP_ZONE_DISTANCE + REENTER_ARM_MARGIN;
                                    if (!worldBEntryArmedRef.current && distToTarget > reenterDistance) {
                                        worldBEntryArmedRef.current = true;
                                    }
                                    if (worldBEntryArmedRef.current && isDeepZone) {
                                        switchWorld('B');
                                    }
                                }

                                if (currentWorldRef.current === 'B' && worldCUnlockedRef.current) {
                                    const reenterDistanceC = DEEP_ZONE_DISTANCE + REENTER_ARM_MARGIN_C;
                                    if (!worldCEntryArmedRef.current && distToTarget > reenterDistanceC) {
                                        worldCEntryArmedRef.current = true;
                                    }
                                    if (worldCEntryArmedRef.current && isDeepZone) {
                                        enterWorldC();
                                    }
                                }

                                if (currentWorldRef.current === 'C' && worldDUnlockedRef.current) {
                                    const reenterDistanceD = DEEP_ZONE_DISTANCE + REENTER_ARM_MARGIN_D;
                                    if (!worldDEntryArmedRef.current && distToTarget > reenterDistanceD) {
                                        worldDEntryArmedRef.current = true;
                                    }
                                    if (worldDEntryArmedRef.current && isDeepZone) {
                                        enterWorldD();
                                    }
                                }

                                if (currentWorldRef.current === 'D') {
                                    const stressActive = worldDStressActiveRef.current;
                                    if (stressActive) {
                                        if (!worldDStressHoldStartRef.current) {
                                            worldDStressHoldStartRef.current = performance.now();
                                        }
                                        const elapsed = performance.now() - worldDStressHoldStartRef.current;
                                        const progress = Math.min(elapsed / WORLD_D_STRESS_HOLD_MS, 1);
                                        if (Math.abs(progress - worldDStressHoldProgressRef.current) > 0.01) {
                                            worldDStressHoldProgressRef.current = progress;
                                            setWorldDStressProgress(progress);
                                        }
                                    }
                                    if (worldDGroupRef.current) {
                                        const pulse = 0.5 + 0.5 * Math.sin(elapsedTime * 6);
                                        const squeeze = stressActive ? 0.86 + 0.05 * Math.sin(elapsedTime * 8) : 1;
                                        const top = worldDTopRef.current;
                                        const bottom = worldDBottomRef.current;
                                        if (top && bottom) {
                                            const baseTop = (top.userData.baseY as number) ?? top.position.y;
                                            const baseBottom = (bottom.userData.baseY as number) ?? bottom.position.y;
                                            top.position.y = baseTop * squeeze;
                                            bottom.position.y = baseBottom * squeeze;
                                            const scaleY = stressActive ? 0.9 + 0.05 * Math.sin(elapsedTime * 9) : 1;
                                            top.scale.set(1, scaleY, 1);
                                            bottom.scale.set(1, scaleY, 1);
                                        }
                                        if (worldDGrooveRef.current) {
                                            worldDGrooveRef.current.scale.set(1, squeeze, 1);
                                        }
                                        const groupScale = stressActive ? 1 + 0.015 * Math.sin(elapsedTime * 6) : 1;
                                        worldDGroupRef.current.scale.setScalar(groupScale);
                                        if (worldDMaterialRef.current && worldDGrooveMaterialRef.current) {
                                            if (stressActive) {
                                                worldDMaterialRef.current.color.setHex(0xfafafa);
                                                worldDMaterialRef.current.emissive.setHex(0x1cffff);
                                                worldDMaterialRef.current.emissiveIntensity = 0.6 + 0.25 * pulse;
                                                worldDGrooveMaterialRef.current.color.setHex(0x1a1a1a);
                                                worldDGrooveMaterialRef.current.emissive.setHex(0x00b3ff);
                                                worldDGrooveMaterialRef.current.emissiveIntensity = 0.5 + 0.2 * pulse;
                                            } else {
                                                worldDMaterialRef.current.color.setHex(0xdedede);
                                                worldDMaterialRef.current.emissive.setHex(0x111111);
                                                worldDMaterialRef.current.emissiveIntensity = 0.4;
                                                worldDGrooveMaterialRef.current.color.setHex(0x8a8a8a);
                                                worldDGrooveMaterialRef.current.emissive.setHex(0x111111);
                                                worldDGrooveMaterialRef.current.emissiveIntensity = 0.4;
                                            }
                                        }
                                    }
                                    if (worldDParticlesRef.current && worldDParticleDataRef.current) {
                                        const { positions, velocities, basePositions } = worldDParticleDataRef.current;
                                        const pull = stressActive ? 0.006 : 0.004;
                                        const relax = 0.02;
                                        const damp = 0.88;
                                        for (let i = 0; i < positions.length; i += 3) {
                                            const px = positions[i];
                                            const py = positions[i + 1];
                                            const pz = positions[i + 2];
                                            let vx = velocities[i];
                                            let vy = velocities[i + 1];
                                            let vz = velocities[i + 2];
                                            if (stressActive) {
                                                vx += -px * pull;
                                                vy += -py * pull;
                                                vz += -pz * pull;
                                            } else {
                                                vx += (basePositions[i] - px) * relax;
                                                vy += (basePositions[i + 1] - py) * relax;
                                                vz += (basePositions[i + 2] - pz) * relax;
                                            }
                                            vx *= damp;
                                            vy *= damp;
                                            vz *= damp;
                                            positions[i] = px + vx;
                                            positions[i + 1] = py + vy;
                                            positions[i + 2] = pz + vz;
                                            velocities[i] = vx;
                                            velocities[i + 1] = vy;
                                            velocities[i + 2] = vz;
                                        }
                                        const geometry = worldDParticlesRef.current.geometry as THREE.BufferGeometry;
                                        (geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
                                        const material = worldDParticlesRef.current.material as THREE.PointsMaterial;
                                        if (stressActive) {
                                            material.color.setHex(0xffffff);
                                            material.opacity = 0.85;
                                        } else {
                                            material.color.setHex(0xaefcff);
                                            material.opacity = 0.6;
                                        }
                                    }
                                }

if (materialRef.current) {
                    materialRef.current.uniforms.uTime.value = elapsedTime;
                    materialRef.current.uniforms.uDisplacementStrength.value = paramsRef.current.displacement;
                    materialRef.current.uniforms.uNoiseSpeed.value = paramsRef.current.noiseSpeed;
                    materialRef.current.uniforms.uPointSize.value = paramsRef.current.pointSize;
                    materialRef.current.uniforms.uThreshold.value = paramsRef.current.threshold;
                    materialRef.current.uniforms.uOpacity.value = paramsRef.current.opacity;
                    
                    if(composerRef.current && composerRef.current.passes[1] instanceof UnrealBloomPass) {
                        (composerRef.current.passes[1] as UnrealBloomPass).strength = paramsRef.current.bloomStrength;
                    }

                    if (animatingRef.current) {
                        const currentGrowth = materialRef.current.uniforms.uGrowth.value;
                        const nextGrowth = Math.min(currentGrowth + paramsRef.current.growthSpeed, MAX_GROWTH);
                        materialRef.current.uniforms.uGrowth.value = nextGrowth;
                        setGrowth(nextGrowth);
                        syncGrowthRef(nextGrowth);
                        if (nextGrowth >= MAX_GROWTH) {
                            animatingRef.current = false;
                            setAnimating(false);
                            if (currentWorldRef.current === 'A') {
                                initialAGrowthCompletedRef.current = true;
                            }
                        }
                    }
                }

                composer.render();
            };
            
            clockRef.current.start();
            animate();
            
            animatingRef.current = true;
            setAnimating(true);
            setGrowth(0);
            setLoading(false);
            const preloadTarget = currentWorldRef.current === 'A' ? worldBImageRef.current : worldAImageRef.current;
            preloadTexture(preloadTarget);
            preloadWorldCModel();
            ensureWorldDModel();
        },
        undefined, // onProgress
        (err) => {
            const isDefaultTexture = textureUrl === DEFAULT_IMAGE || textureUrl === WORLD_B_IMAGE;
            if (isDefaultTexture) {
                 console.warn(`Default image at ${DEFAULT_IMAGE} failed to load. Falling back.`);
                 if (currentWorldRef.current === 'A') {
                     worldAImageRef.current = FALLBACK_IMAGE;
                 } else {
                     worldBImageRef.current = FALLBACK_IMAGE;
                 }
                 setImageSrc(FALLBACK_IMAGE);
                 return;
            }

            console.error("Texture Load Error:", err);
            setError("Could not load image. Please try a valid image file or URL.");
            setLoading(false);
            setImageSrc(null); // Reset to upload screen
        }
    );

    const handleResize = () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        if(composerRef.current) composerRef.current.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
        window.removeEventListener('resize', handleResize);
    };

  }, [buildGeometryFromTexture, preloadTexture, preloadWorldCModel, ensureWorldDModel]);

  useEffect(() => {
     if (!imageSrc) {
         setLoading(false);
     } else {
         setLoading(true);
         setError(null);
         const t = setTimeout(() => initScene(imageSrc), 0);
         return () => clearTimeout(t);
     }
     return cleanupScene;
  }, [imageSrc, initScene]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setLoading(true);
      setError(null);
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
            setWorldImage(event.target.result as string);
        }
      };
      reader.onerror = () => {
          setError("Failed to read file.");
          setLoading(false);
      }
      reader.readAsDataURL(file);
    }
  };

  const restartAnimation = () => {
      if(materialRef.current) {
          materialRef.current.uniforms.uGrowth.value = 0.0;
          setGrowth(0);
          syncGrowthRef(0);
          if (currentWorldRef.current === 'A') {
              initialAGrowthCompletedRef.current = false;
          }
          animatingRef.current = true;
          setAnimating(true);
      }
  };

  const handleGrowthSlider = (val: number) => {
      animatingRef.current = false;
      setAnimating(false);
      setGrowth(val);
      syncGrowthRef(val);
      if(materialRef.current) {
          materialRef.current.uniforms.uGrowth.value = val;
      }
  };

  const toggleHandControl = () => {
      if (handControlEnabled) {
          setHandControlEnabled(false);
          // Re-enable auto rotate when hands are off
          if (controlsRef.current) controlsRef.current.autoRotate = true;
      } else {
          setHandControlEnabled(true);
          // Disable auto rotate when hands are on
          if (controlsRef.current) controlsRef.current.autoRotate = false;
      }
  };

  const growthPercentage = Math.min(100, Math.floor((growth / MAX_GROWTH) * 100));

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden font-sans text-white">
      <div ref={mountRef} className="absolute inset-0 z-0" />

      {/* Intro / Empty State */}
      {!imageSrc && !loading && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80 backdrop-blur-md">
          <div className="text-center max-w-lg p-8 border border-neutral-800 rounded-2xl bg-neutral-900/50 shadow-2xl">
            <h1 className="text-3xl font-light tracking-[0.2em] mb-4 text-emerald-400">PHYSARUM</h1>
            <p className="text-neutral-400 mb-8 font-light">
              True Color 3D Particle System.<br/>
              Simulates organic slime mold growth.
            </p>
            
            {error && (
                <div className="mb-6 p-3 bg-red-900/30 border border-red-800/50 rounded-lg flex items-center justify-center gap-2 text-red-200 text-sm">
                    <AlertCircle className="w-4 h-4" />
                    <span>{error}</span>
                </div>
            )}

            <label className="group relative inline-flex items-center justify-center px-8 py-3 text-sm font-medium text-black transition-all duration-200 bg-white rounded-full cursor-pointer hover:bg-neutral-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-neutral-900">
              <Upload className="w-4 h-4 mr-2" />
              <span>Upload Image</span>
              <input type="file" className="hidden" accept="image/*" onChange={handleFileUpload} />
            </label>
            <div className="mt-4">
                 <button onClick={handleDemoImage} className="text-xs text-neutral-500 hover:text-white underline">
                     Try Demo Image
                 </button>
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-emerald-500 mb-4"></div>
          <p className="text-neutral-500 text-xs tracking-widest animate-pulse">GENERATING PARTICLES</p>
        </div>
      )}

      {/* Video Element for MediaPipe (Hidden but functional) */}
      <video 
        ref={videoRef} 
        className={`absolute top-24 right-6 w-32 h-24 rounded-lg object-cover border-2 border-emerald-500/50 z-20 ${handControlEnabled ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}
        style={{ transform: 'scaleX(-1)' }}
        playsInline 
      />

      {/* Main Controls */}
      {imageSrc && !loading && (
        <>
            <div className="absolute top-0 left-0 w-full p-6 flex justify-between items-start z-10 pointer-events-none">
                <div>
                    <h2 className="text-xl font-light tracking-widest text-white/80">PHYSARUM</h2>
                    <p className="text-xs text-emerald-500/80 font-mono mt-1">PARTICLE: {growthPercentage}% GROWTH</p>
                </div>
                
                <div className="flex gap-4 pointer-events-auto items-center">
                    <button 
                        onClick={toggleHandControl}
                        className={`p-3 border rounded-full text-white transition-colors ${handControlEnabled ? 'bg-emerald-900/80 border-emerald-500' : 'bg-neutral-900/80 border-neutral-800 hover:border-emerald-500'}`}
                        title={handControlEnabled ? "Disable Hand Control" : "Enable Hand Control"}
                    >
                        {handControlEnabled ? <Video className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
                    </button>
                    <button 
                        onClick={restartAnimation}
                        className="p-3 bg-neutral-900/80 border border-neutral-800 rounded-full text-white hover:bg-emerald-900/50 hover:border-emerald-500 transition-colors"
                        title="Restart Growth"
                    >
                        <RefreshCcw className="w-5 h-5" />
                    </button>
                    <label className="p-3 bg-neutral-900/80 border border-neutral-800 rounded-full text-white hover:bg-neutral-800 cursor-pointer transition-colors" title="Change Image">
                        <Camera className="w-5 h-5" />
                        <input type="file" className="hidden" accept="image/*" onChange={handleFileUpload} />
                    </label>
                </div>
            </div>

            <div className="absolute bottom-6 left-6 z-10 w-80 pointer-events-auto">
                <div className="bg-neutral-950/80 backdrop-blur-md border border-neutral-800 rounded-xl p-6 shadow-2xl">
                    <div className="flex items-center gap-2 mb-4 text-emerald-400 border-b border-neutral-800 pb-2">
                        <span className="text-xs font-bold tracking-widest uppercase">Mission</span>
                    </div>

                    <div className="space-y-4 text-xs text-neutral-300">
                        <div className="flex items-start gap-3">
                            <span className={`mt-1 inline-block h-2 w-2 rounded-full ${deepZoneReached ? 'bg-emerald-500' : 'bg-neutral-700'}`} />
                            <div className="flex-1">
                                <div className={`text-[11px] ${deepZoneReached ? 'text-emerald-300' : 'text-neutral-300'}`}>
                                    Task 1: Enter the deep zone
                                </div>
                                <div className="text-[10px] text-neutral-500">Zoom into the deepest area to unlock the gate.</div>
                            </div>
                        </div>

                        <div className="flex items-start gap-3">
                            <span className={`mt-1 inline-block h-2 w-2 rounded-full ${worldBUnlocked ? 'bg-emerald-500' : 'bg-neutral-700'}`} />
                            <div className="flex-1">
                                <div className={`text-[11px] ${worldBUnlocked ? 'text-emerald-300' : 'text-neutral-300'}`}>
                                    Task 2: Left-hand OK for 3 seconds
                                </div>
                                {!worldBUnlocked && (
                                    <div className="mt-2">
                                        <div className="flex justify-between text-[10px] text-neutral-500 mb-1">
                                            <span>Confirm progress</span>
                                            <span>{Math.round(okHoldProgress * 100)}%</span>
                                        </div>
                                        <div className="h-1 w-full bg-neutral-800 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-emerald-500 transition-[width] duration-150"
                                                style={{ width: `${Math.round(okHoldProgress * 100)}%` }}
                                            />
                                        </div>
                                    </div>
                                )}
                                {worldBUnlocked && (
                                    <div className="text-[10px] text-emerald-400 mt-1">World B unlocked</div>
                                )}
                            </div>
                        </div>

                        {worldBUnlocked && (
                            <>
                                <div className="flex items-start gap-3">
                                    <span className={`mt-1 inline-block h-2 w-2 rounded-full ${deepZoneReachedB ? 'bg-emerald-500' : 'bg-neutral-700'}`} />
                                    <div className="flex-1">
                                        <div className={`text-[11px] ${deepZoneReachedB ? 'text-emerald-300' : 'text-neutral-300'}`}>
                                            Task 3: Enter World B deep zone
                                        </div>
                                        <div className="text-[10px] text-neutral-500">Zoom into World B to unlock the next gate.</div>
                                    </div>
                                </div>

                            </>
                        )}

                                                {worldCUnlocked && (
                            <>
                                <div className="flex items-start gap-3">
                                    <span className={`mt-1 inline-block h-2 w-2 rounded-full ${deepZoneReachedC ? 'bg-emerald-500' : 'bg-neutral-700'}`} />
                                    <div className="flex-1">
                                        <div className={`text-[11px] ${deepZoneReachedC ? 'text-emerald-300' : 'text-neutral-300'}`}>
                                            Task 4: Enter World C deep zone
                                        </div>
                                        <div className="text-[10px] text-neutral-500">Zoom into World C to unlock the next gate.</div>
                                    </div>
                                </div>

                                <div className="flex items-start gap-3">
                                    <span className={`mt-1 inline-block h-2 w-2 rounded-full ${worldDUnlocked ? 'bg-emerald-500' : 'bg-neutral-700'}`} />
                                    <div className="flex-1">
                                        <div className={`text-[11px] ${worldDUnlocked ? 'text-emerald-300' : 'text-neutral-300'}`}>
                                            Task 5: Left-hand OK for 3 seconds
                                        </div>
                                        {!worldDUnlocked && (
                                            <div className="mt-2">
                                                <div className="flex justify-between text-[10px] text-neutral-500 mb-1">
                                                    <span>Confirm progress</span>
                                                    <span>{Math.round(okHoldProgressD * 100)}%</span>
                                                </div>
                                                <div className="h-1 w-full bg-neutral-800 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-emerald-500 transition-[width] duration-150"
                                                        style={{ width: `${Math.round(okHoldProgressD * 100)}%` }}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                        {worldDUnlocked && (
                                            <div className="text-[10px] text-emerald-400 mt-1">World D unlocked</div>
                                        )}
                                    </div>
                                </div>
                            </>
                        )}

                        {currentWorld === 'D' && (
                            <>
                                <div className="text-[10px] text-neutral-500 uppercase tracking-widest">Goal</div>
                                <div className="text-[10px] text-neutral-400">Observe bacterial division under mechanical stress.</div>
                                <div className="flex items-start gap-3">
                                    <span className={`mt-1 inline-block h-2 w-2 rounded-full ${worldDRotationProgress >= 1 ? 'bg-emerald-500' : 'bg-neutral-700'}`} />
                                    <div className="flex-1">
                                        <div className={`text-[11px] ${worldDRotationProgress >= 1 ? 'text-emerald-300' : 'text-neutral-300'}`}>
                                            Task 6: Left-hand wrist rotation &gt;= 1.5 rad
                                        </div>
                                        {worldDRotationProgress < 1 && (
                                            <div className="mt-2">
                                                <div className="flex justify-between text-[10px] text-neutral-500 mb-1">
                                                    <span>Rotation progress</span>
                                                    <span>{Math.round(worldDRotationProgress * 100)}%</span>
                                                </div>
                                                <div className="h-1 w-full bg-neutral-800 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-emerald-500 transition-[width] duration-150"
                                                        style={{ width: `${Math.round(worldDRotationProgress * 100)}%` }}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                        {worldDRotationProgress >= 1 && (
                                            <div className="text-[10px] text-emerald-400 mt-1">Rotation confirmed</div>
                                        )}
                                    </div>
                                </div>

                                <div className="flex items-start gap-3">
                                    <span className={`mt-1 inline-block h-2 w-2 rounded-full ${worldDStressProgress >= 1 ? 'bg-emerald-500' : 'bg-neutral-700'}`} />
                                    <div className="flex-1">
                                        <div className={`text-[11px] ${worldDStressProgress >= 1 ? 'text-emerald-300' : 'text-neutral-300'}`}>
                                            Task 7: Left-hand fist stress for 3 seconds
                                        </div>
                                        {worldDStressProgress < 1 && (
                                            <div className="mt-2">
                                                <div className="flex justify-between text-[10px] text-neutral-500 mb-1">
                                                    <span>{worldDStressActive ? 'Stress active' : 'Awaiting left-hand fist'}</span>
                                                    <span>{Math.round(worldDStressProgress * 100)}%</span>
                                                </div>
                                                <div className="h-1 w-full bg-neutral-800 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-emerald-500 transition-[width] duration-150"
                                                        style={{ width: `${Math.round(worldDStressProgress * 100)}%` }}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                        {worldDStressProgress >= 1 && (
                                            <div className="text-[10px] text-emerald-400 mt-1">Stress hold confirmed</div>
                                        )}
                                    </div>
                                </div>
                            </>
                        )}

                        {!worldBUnlocked && !handControlEnabled && (
                            <div className="text-[10px] text-neutral-500">Tip: enable hand control to confirm.</div>
                        )}
                        {worldBUnlocked && !worldCUnlocked && currentWorld === 'B' && (
                            <div className="text-[10px] text-neutral-500">Tip: zoom into World B deep zone to enter World C.</div>
                        )}
                        {worldCUnlocked && !worldDUnlocked && !handControlEnabled && (
                            <div className="text-[10px] text-neutral-500">Tip: keep hand control on to confirm Task 5.</div>
                        )}
                        {currentWorld === 'B' && (
                            <div className="text-[10px] text-neutral-500">Tip: zoom out to return to World A{worldCUnlocked ? ', zoom in to enter World C.' : '.'}</div>
                        )}
                        {currentWorld === 'C' && (
                            <div className="text-[10px] text-neutral-500">Tip: zoom out to return to World B{worldDUnlocked ? ', zoom in to enter World D.' : '.'}</div>
                        )}
                        {currentWorld === 'D' && (
                            <div className="text-[10px] text-neutral-500">Tip: zoom out to return to World C.</div>
                        )}
                    </div>
                </div>
            </div>

            <div className="absolute bottom-6 right-6 z-10 w-80 pointer-events-auto">
                <div className="bg-neutral-950/80 backdrop-blur-md border border-neutral-800 rounded-xl p-6 shadow-2xl">
                    <div className="flex items-center gap-2 mb-4 text-emerald-400 border-b border-neutral-800 pb-2">
                        <Settings className="w-4 h-4" />
                        <span className="text-xs font-bold tracking-widest uppercase">Configuration</span>
                    </div>

                    <div className="space-y-5">
                         {handControlEnabled && (
                             <div className="bg-emerald-900/20 p-2 rounded border border-emerald-500/20 text-xs text-emerald-200 mb-2 flex flex-col gap-1">
                                <div className="flex items-center gap-2">
                                    <Hand className="w-3 h-3" />
                                    <span className="font-bold">Right Hand Active</span>
                                </div>
                                <span className="opacity-70 pl-5">Move hand to pan</span>
                                <span className="opacity-70 pl-5">👌Pinch close: Zoom In</span>
                                <span className="opacity-70 pl-5">🖐Pinch open: Zoom Out</span>
                             </div>
                         )}

                        <div className="space-y-1">
                            <div className="flex justify-between text-xs text-neutral-400">
                                <span>Growth Progress</span>
                                <span>{growthPercentage}%</span>
                            </div>
                            <input 
                                type="range" 
                                min="0" 
                                max={MAX_GROWTH} 
                                step="0.01"
                                value={growth} 
                                onChange={(e) => handleGrowthSlider(parseFloat(e.target.value))}
                                className="w-full h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                            />
                        </div>

                        <ControlSlider label="Displacement" min={0} max={300} initial={80} onChange={(v) => paramsRef.current.displacement = v} />
                        <ControlSlider label="Flow Speed" min={0} max={2} step={0.1} initial={0.2} onChange={(v) => paramsRef.current.noiseSpeed = v} />
                        <ControlSlider label="Particle Size" min={1} max={10} step={0.1} initial={3.5} onChange={(v) => paramsRef.current.pointSize = v} />
                        <ControlSlider label="Bloom Strength" min={0} max={3} step={0.1} initial={0.3} onChange={(v) => paramsRef.current.bloomStrength = v} />
                        <ControlSlider label="Threshold" min={0} max={0.5} step={0.01} initial={0.15} onChange={(v) => paramsRef.current.threshold = v} />
                    </div>
                </div>
            </div>
        </>
      )}
    </div>
  );
};

const ControlSlider: React.FC<{
    label: string; 
    min: number; 
    max: number; 
    step?: number; 
    initial: number; 
    onChange: (val: number) => void; 
}> = ({ label, min, max, step = 1, initial, onChange }) => {
    const [val, setVal] = useState(initial);
    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs text-neutral-500 font-mono">
                <span>{label}</span>
                <span>{val}</span>
            </div>
            <input 
                type="range" 
                min={min} 
                max={max} 
                step={step} 
                value={val} 
                onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setVal(v);
                    onChange(v);
                }}
                className="w-full h-1 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-white hover:accent-emerald-400"
            />
        </div>
    );
};

