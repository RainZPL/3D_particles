import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
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

// Fallback in case A.png is missing
const FALLBACK_IMAGE = "https://images.unsplash.com/photo-1563089145-599997674d42?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80";
const MAX_GROWTH = 0.85; 

// Zoom Thresholds
const ZOOM_IN_THRESHOLD = 0.05; // Fingers close together
const ZOOM_OUT_THRESHOLD = 0.12; // Fingers spread apart

export function App() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(DEFAULT_IMAGE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [animating, setAnimating] = useState(false);
  const animatingRef = useRef(false); 
  const [growth, setGrowth] = useState(0); 
  
  // Hand Gesture State
  const [handControlEnabled, setHandControlEnabled] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const previousHandPos = useRef<{x: number, y: number} | null>(null);
  const smoothedHandPos = useRef<{x: number, y: number} | null>(null);
  const handVelocity = useRef<{x: number, y: number}>({ x: 0, y: 0 });
  const lastHandTs = useRef<number | null>(null);
  // Refs for Three.js objects
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const clockRef = useRef<THREE.Clock>(new THREE.Clock());
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

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

  // --- Hand Tracking Logic ---

  useEffect(() => {
    if (!handControlEnabled || !videoRef.current) return;

    let handsInstance: any = null;
    let cameraInstance: any = null;

    const onResults = (results: Results) => {
        if (!controlsRef.current || !cameraRef.current) return;

        // 1. Filter for Right Hand Only
        let rightHandIndex = -1;
        if (results.multiHandedness) {
            for (let i = 0; i < results.multiHandedness.length; i++) {
                // In selfieMode=true, "Right" label corresponds to the user's actual Right hand
                if (results.multiHandedness[i].label === 'Right') {
                    rightHandIndex = i;
                    break;
                }
            }
        }

        const landmarks = results.multiHandLandmarks;

        if (rightHandIndex !== -1 && landmarks && landmarks[rightHandIndex]) {
            const hand = landmarks[rightHandIndex];
            
            // --- Panning Logic (Wrist) ---
            const rawX = hand[0].x;
            const rawY = hand[0].y;

            if (!smoothedHandPos.current) {
                smoothedHandPos.current = { x: rawX, y: rawY };
            } else {
                // Low-pass filter to reduce jitter.
                smoothedHandPos.current = {
                    x: THREE.MathUtils.lerp(smoothedHandPos.current.x, rawX, 0.25),
                    y: THREE.MathUtils.lerp(smoothedHandPos.current.y, rawY, 0.25),
                };
            }

            if (previousHandPos.current) {
                // Sensitivity factor
                const sensitivity = 200; 
                const now = performance.now();
                const dt = lastHandTs.current ? Math.min((now - lastHandTs.current) / 1000, 0.05) : 1 / 60;
                lastHandTs.current = now;
                
                // Calculate deltas
                // Note: In selfie mode, x moves normal (left is 0, right is 1)
                const deltaX = (smoothedHandPos.current.x - previousHandPos.current.x) * sensitivity;
                const deltaY = (smoothedHandPos.current.y - previousHandPos.current.y) * sensitivity;
                const deadZone = 0.25;
                const targetX = Math.abs(deltaX) < deadZone ? 0 : deltaX;
                const targetY = Math.abs(deltaY) < deadZone ? 0 : deltaY;
                const smooth = 0.85;
                handVelocity.current.x = THREE.MathUtils.lerp(handVelocity.current.x, targetX, 1 - smooth);
                handVelocity.current.y = THREE.MathUtils.lerp(handVelocity.current.y, targetY, 1 - smooth);
                const speed = Math.min(dt * 60, 2);

                const cameraObj = cameraRef.current;
                
                // Get Camera vectors
                const right = new THREE.Vector3(1, 0, 0).applyQuaternion(cameraObj.quaternion);
                const up = new THREE.Vector3(0, 1, 0).applyQuaternion(cameraObj.quaternion);

                // "Camera follows hand" logic
                // Hand Right (+X) -> Camera moves Right (+RightVector)
                // Hand Down (+Y in MP) -> Camera moves Down (-UpVector)
                
                const panVector = new THREE.Vector3()
                    .addScaledVector(right, handVelocity.current.x * speed)
                    .addScaledVector(up, -handVelocity.current.y * speed); 
                
                cameraObj.position.add(panVector);
                controlsRef.current.target.add(panVector);
            }

            previousHandPos.current = { x: smoothedHandPos.current.x, y: smoothedHandPos.current.y };
            // --- Continuous Zoom Logic (Thumb Tip 4 to Index Tip 8) ---
            const thumb = hand[4];
            const index = hand[8];
            const dist = Math.sqrt(
                Math.pow(thumb.x - index.x, 2) + 
                Math.pow(thumb.y - index.y, 2)
            );

            // Determine Zoom State based on Thresholds
            const zoomSpeed = 5.0; // Speed of continuous zoom
            const cameraObj = cameraRef.current;
            const viewDirection = new THREE.Vector3();
            cameraObj.getWorldDirection(viewDirection);

            // Distance Check to prevent clipping or getting lost
            const distToTarget = cameraObj.position.distanceTo(controlsRef.current.target);

            if (dist < ZOOM_IN_THRESHOLD) {
                // Close pinch = Zoom In (Move forward)
                if (distToTarget > 50) { // Min distance limit
                     cameraObj.position.addScaledVector(viewDirection, zoomSpeed);
                }
            } else if (dist > ZOOM_OUT_THRESHOLD) {
                // Open spread = Zoom Out (Move backward)
                if (distToTarget < 2000) { // Max distance limit
                    cameraObj.position.addScaledVector(viewDirection, -zoomSpeed);
                }
            }
            
            controlsRef.current.update();

        } else {
            // Reset if hand lost or wrong hand
            previousHandPos.current = null;
            smoothedHandPos.current = null;
            handVelocity.current = { x: 0, y: 0 };
            lastHandTs.current = null;
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
                        if (videoRef.current && handsInstance) {
                            await handsInstance.send({ image: videoRef.current });
                        }
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
        if (cameraInstance) cameraInstance.stop();
        if (handsInstance) handsInstance.close();
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

    // Texture Loading
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = 'anonymous';
    
    loader.load(
        textureUrl, 
        (texture) => {
            // Success Callback
            const imgWidth = texture.image.width;
            const imgHeight = texture.image.height;
            
            const maxDim = 600; 
            let rw = imgWidth, rh = imgHeight;
            if(imgWidth > maxDim || imgHeight > maxDim) {
                const aspect = imgWidth / imgHeight;
                if(imgWidth > imgHeight) { rw = maxDim; rh = maxDim / aspect; }
                else { rh = maxDim; rw = maxDim * aspect; }
            }

            const geometry = new THREE.PlaneGeometry(rw, rh, Math.floor(rw), Math.floor(rh));

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

            // Animation Loop
            const animate = () => {
                animationFrameRef.current = requestAnimationFrame(animate);
                
                const elapsedTime = clockRef.current.getElapsedTime();
                
                // Only auto-rotate if hands aren't controlling it to avoid fighting
                // Also update damping
                controls.update();

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
                        if (currentGrowth < MAX_GROWTH) { 
                            materialRef.current.uniforms.uGrowth.value += paramsRef.current.growthSpeed;
                            setGrowth(materialRef.current.uniforms.uGrowth.value);
                        } else {
                            animatingRef.current = false;
                            setAnimating(false);
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
        },
        undefined, // onProgress
        (err) => {
            if (textureUrl === DEFAULT_IMAGE) {
                 console.warn(`Default image at ${DEFAULT_IMAGE} failed to load. Falling back.`);
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

  }, []);

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
            setImageSrc(event.target.result as string);
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
          animatingRef.current = true;
          setAnimating(true);
      }
  };

  const handleGrowthSlider = (val: number) => {
      animatingRef.current = false;
      setAnimating(false);
      setGrowth(val);
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
                 <button onClick={() => setImageSrc(DEFAULT_IMAGE)} className="text-xs text-neutral-500 hover:text-white underline">
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
                                <span className="opacity-70 pl-5">â€?Move hand to pan</span>
                                <span className="opacity-70 pl-5">â€?Pinch close: Zoom In</span>
                                <span className="opacity-70 pl-5">â€?Pinch open: Zoom Out</span>
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

