import React, { useEffect, useRef } from 'react';

/**
 * VignetteBloomCanvas - Canvas2D ASCII & Raster Primitive Art Renderer
 * Recreates the "Vignette Bloom" effect pipeline from 21st.dev.
 */

export const DEFAULT_VIGNETTE_PARAMS = {
  renderMode: "mosaic",
  bgMode: "solid",
  bgBlur: 12,
  bgOpacity: 90,
  cellSize: 16,
  coverage: 100,
  invert: false,
  styleBlend: "source-over",
  charSet: "standard",
  customChars: "",
  brightness: 12,
  contrast: 115,
  edgeEmphasis: 0,
  density: 0,
  toneCurve: [
    { x: 0, y: 0 },
    { x: 1, y: 1 }
  ],
  tint: "#3ca6ff",
  tintOpacity: 0,
  overlayBlend: "multiply",
  saturation: 100,
  grayscale: 0,
  blurType: "off",
  blurAmount: 35,
  blurAngle: 0,
  directionalBothSides: false,
  tiltFocus: 35,
  tiltPosition: 50,
  tiltFeather: 15,
  lensFocus: 40,
  blurCenterX: 50,
  blurCenterY: 50,
  progressivePosition: 55,
  progressiveReverse: false,
  pfx: {
    vignette: { enabled: true, intensity: 38 },
    scanLines: { enabled: false, intensity: 40 },
    chromatic: { enabled: false, intensity: 15 },
    bloom: { enabled: true, intensity: 25 },
    filmGrain: { enabled: false, intensity: 30 },
    glitch: { enabled: false, intensity: 20 },
    pixelate: { enabled: false, intensity: 15 },
    halftone: { enabled: false, intensity: 20 },
    filmDust: { enabled: false, intensity: 20 }
  },
  animated: true,
  animStyle: "wave",
  animSpeed: { enabled: true, intensity: 100 },
  animIntensity: { enabled: true, intensity: 60 },
  lights: { enabled: false, points: [] },
  mask: {
    enabled: false,
    tool: "freehand",
    brushSize: 30,
    showOverlay: false,
    invert: false,
    dataUrl: null,
    shapes: []
  }
};

const CHAR_SETS = {
  standard: " .:-=+*#%@@",
  blocks: " ░▒▓█",
  minimal: " .+#@",
  binary: "01",
  matrix: "0123456789ABCDEFｦｱｳｴｵｶｷｹｺｻｼｽｾｿﾀﾂﾃﾅﾆﾇﾈﾊﾋﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾜ",
  hexdump: "0123456789ABCDEF",
  braille: "⠀⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟"
};

export default function VignetteBloomCanvas({
  config = DEFAULT_VIGNETTE_PARAMS,
  imageSrc = null,
  width = 800,
  height = 500,
  className = ""
}) {
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(document.createElement('canvas'));
  const animFrameRef = useRef(null);
  const imageObjRef = useRef(null);

  // Merge provided config with defaults
  const params = {
    ...DEFAULT_VIGNETTE_PARAMS,
    ...config,
    pfx: { ...DEFAULT_VIGNETTE_PARAMS.pfx, ...(config.pfx || {}) },
    animSpeed: { ...DEFAULT_VIGNETTE_PARAMS.animSpeed, ...(config.animSpeed || {}) },
    animIntensity: { ...DEFAULT_VIGNETTE_PARAMS.animIntensity, ...(config.animIntensity || {}) },
  };

  // Generate procedural satellite/tech texture if image is missing or loading fails
  const generateProceduralTexture = (ctx, w, h) => {
    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, '#0a192f');
    grad.addColorStop(0.3, '#102a45');
    grad.addColorStop(0.7, '#1b4965');
    grad.addColorStop(1, '#0b132b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Draw tech grid & satellite landmass contours
    ctx.fillStyle = '#3ca6ff';
    for (let i = 0; i < 40; i++) {
      const cx = (Math.sin(i * 99) * 0.5 + 0.5) * w;
      const cy = (Math.cos(i * 33) * 0.5 + 0.5) * h;
      const r = (Math.sin(i * 12) * 0.5 + 0.5) * 80 + 20;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.globalAlpha = 0.15;
      ctx.fill();
    }
    ctx.globalAlpha = 1.0;
  };

  useEffect(() => {
    // Load image
    const img = new Image();
    img.crossOrigin = "anonymous";
    const srcToUse = imageSrc || "/ascii-editor/demos/generated/ref-008.webp";

    img.onload = () => {
      imageObjRef.current = img;
    };
    img.onerror = () => {
      // Fall back to built-in futuristic satellite reference
      imageObjRef.current = null;
    };
    img.src = srcToUse;

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [imageSrc]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    let startTime = performance.now();

    const renderFrame = (timestamp) => {
      const elapsed = (timestamp - startTime) / 1000;
      const w = width;
      const h = height;

      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;

      const offCanvas = offscreenCanvasRef.current;
      offCanvas.width = w;
      offCanvas.height = h;
      const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });

      // ────────────────────────────────────────────────────────────────
      // PIPELINE STEP 1: Background & Source Draw
      // ────────────────────────────────────────────────────────────────
      offCtx.clearRect(0, 0, w, h);

      if (params.bgMode === "solid") {
        offCtx.fillStyle = "#090d16";
        offCtx.fillRect(0, 0, w, h);
      } else if (params.bgMode === "blurred" || params.bgMode === "photo") {
        if (imageObjRef.current) {
          offCtx.globalAlpha = params.bgOpacity / 100;
          if (params.bgMode === "blurred") offCtx.filter = `blur(${params.bgBlur}px)`;
          offCtx.drawImage(imageObjRef.current, 0, 0, w, h);
          offCtx.filter = "none";
          offCtx.globalAlpha = 1.0;
        } else {
          generateProceduralTexture(offCtx, w, h);
        }
      }

      // Draw source photo for sampling onto temporary canvas
      const sampleCanvas = document.createElement('canvas');
      sampleCanvas.width = w;
      sampleCanvas.height = h;
      const sampleCtx = sampleCanvas.getContext('2d', { willReadFrequently: true });
      if (imageObjRef.current) {
        sampleCtx.drawImage(imageObjRef.current, 0, 0, w, h);
      } else {
        generateProceduralTexture(sampleCtx, w, h);
      }

      const imgData = sampleCtx.getImageData(0, 0, w, h);
      const data = imgData.data;

      // ────────────────────────────────────────────────────────────────
      // PIPELINE STEP 2 & 3: Divide Grid, Sample Color/Luminance & Draw Primitives
      // ────────────────────────────────────────────────────────────────
      const cellSize = Math.max(4, params.cellSize || 16);
      const cols = Math.ceil(w / cellSize);
      const rows = Math.ceil(h / cellSize);

      const animSpeedVal = params.animSpeed.enabled ? (params.animSpeed.intensity / 50) : 1;
      const animIntVal = params.animIntensity.enabled ? (params.animIntensity.intensity / 100) : 0.5;

      const charSetString = CHAR_SETS[params.charSet] || CHAR_SETS.standard;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          // Coverage filter check
          const cellIndex = r * cols + c;
          if (params.coverage < 100 && (cellIndex * 17) % 100 >= params.coverage) {
            continue;
          }

          const x = c * cellSize;
          const y = r * cellSize;

          // Sample average color / luminance in cell
          let sumR = 0, sumG = 0, sumB = 0, count = 0;
          const stepX = Math.max(1, Math.floor(cellSize / 3));
          const stepY = Math.max(1, Math.floor(cellSize / 3));

          for (let cy = y; cy < Math.min(y + cellSize, h); cy += stepY) {
            for (let cx = x; cx < Math.min(x + cellSize, w); cx += stepX) {
              const idx = (cy * w + cx) * 4;
              sumR += data[idx];
              sumG += data[idx + 1];
              sumB += data[idx + 2];
              count++;
            }
          }

          if (count === 0) count = 1;
          let avgR = sumR / count;
          let avgG = sumG / count;
          let avgB = sumB / count;

          // Luminance calculation
          let lum = (0.299 * avgR + 0.587 * avgG + 0.114 * avgB) / 255;
          if (params.invert) lum = 1 - lum;

          // Apply edge emphasis if enabled
          if (params.edgeEmphasis > 0) {
            const isEdge = (c + r) % 2 === 0 ? 1.2 : 0.8;
            lum = Math.min(1, Math.max(0, lum * isEdge));
          }

          // PIPELINE STEP 8: Animation Movement (Wave, Pulse, Shimmer, Ripple, Flicker)
          let animOffset = 0;
          if (params.animated) {
            const timeT = elapsed * animSpeedVal;
            if (params.animStyle === "wave") {
              animOffset = Math.sin(timeT * 2 + (x * 0.01) + (y * 0.01)) * animIntVal;
            } else if (params.animStyle === "pulse") {
              animOffset = Math.cos(timeT * 3) * animIntVal;
            } else if (params.animStyle === "shimmer") {
              animOffset = Math.sin(timeT * 4 + cellIndex) * animIntVal * 0.5;
            } else if (params.animStyle === "ripple") {
              const dist = Math.hypot(x - w / 2, y - h / 2);
              animOffset = Math.sin(dist * 0.05 - timeT * 4) * animIntVal;
            } else if (params.animStyle === "flicker") {
              animOffset = (Math.random() - 0.5) * animIntVal;
            }
          }

          const adjustedLum = Math.min(1, Math.max(0, lum + animOffset * 0.25));

          // Draw Primitive according to renderMode
          const mode = params.renderMode || "mosaic";
          const centerX = x + cellSize / 2;
          const centerY = y + cellSize / 2;

          offCtx.fillStyle = `rgb(${Math.round(avgR)}, ${Math.round(avgG)}, ${Math.round(avgB)})`;
          offCtx.strokeStyle = `rgb(${Math.round(avgR)}, ${Math.round(avgG)}, ${Math.round(avgB)})`;

          if (mode === "characters") {
            const glyphIdx = Math.floor(adjustedLum * (charSetString.length - 1));
            const char = charSetString[glyphIdx] || " ";
            offCtx.font = `${Math.floor(cellSize * 0.95)}px monospace`;
            offCtx.textAlign = "center";
            offCtx.textBaseline = "middle";
            offCtx.fillText(char, centerX, centerY);
          } else if (mode === "matrix" || mode === "hexdump") {
            const charset = mode === "matrix" ? CHAR_SETS.matrix : CHAR_SETS.hexdump;
            const charIdx = (Math.floor(adjustedLum * 16) + Math.floor(elapsed * 10 + cellIndex)) % charset.length;
            offCtx.font = `bold ${Math.floor(cellSize * 0.9)}px monospace`;
            offCtx.fillStyle = mode === "matrix" ? `rgba(60, 240, 140, ${0.4 + adjustedLum * 0.6})` : offCtx.fillStyle;
            offCtx.textAlign = "center";
            offCtx.textBaseline = "middle";
            offCtx.fillText(charset[charIdx], centerX, centerY);
          } else if (mode === "dots" || mode === "bubbles") {
            const radius = (cellSize / 2) * adjustedLum;
            offCtx.beginPath();
            offCtx.arc(centerX, centerY, Math.max(1, radius), 0, Math.PI * 2);
            offCtx.fill();
          } else if (mode === "hexagons") {
            const rHex = (cellSize / 2) * (0.5 + adjustedLum * 0.5);
            offCtx.beginPath();
            for (let a = 0; a < 6; a++) {
              const angle = (Math.PI / 3) * a;
              const hx = centerX + rHex * Math.cos(angle);
              const hy = centerY + rHex * Math.sin(angle);
              if (a === 0) offCtx.moveTo(hx, hy);
              else offCtx.lineTo(hx, hy);
            }
            offCtx.closePath();
            offCtx.fill();
          } else if (mode === "triangles") {
            offCtx.beginPath();
            offCtx.moveTo(centerX, y + 2);
            offCtx.lineTo(x + cellSize - 2, y + cellSize - 2);
            offCtx.lineTo(x + 2, y + cellSize - 2);
            offCtx.closePath();
            offCtx.fill();
          } else if (mode === "lines" || mode === "diagonal" || mode === "hatch") {
            offCtx.lineWidth = Math.max(1, cellSize * 0.2 * adjustedLum);
            offCtx.beginPath();
            offCtx.moveTo(x, y);
            offCtx.lineTo(x + cellSize, y + cellSize);
            if (mode === "hatch" && adjustedLum > 0.5) {
              offCtx.moveTo(x + cellSize, y);
              offCtx.lineTo(x, y + cellSize);
            }
            offCtx.stroke();
          } else if (mode === "cross" || mode === "plus") {
            offCtx.lineWidth = Math.max(1, cellSize * 0.15);
            offCtx.beginPath();
            offCtx.moveTo(centerX, y + 2);
            offCtx.lineTo(centerX, y + cellSize - 2);
            offCtx.moveTo(x + 2, centerY);
            offCtx.lineTo(x + cellSize - 2, centerY);
            offCtx.stroke();
          } else {
            // "mosaic", "pixel", "dither", "lego", "halfblocks", default
            const pad = mode === "lego" ? 1 : 0;
            const size = Math.max(1, cellSize - pad * 2);
            offCtx.fillRect(x + pad, y + pad, size, size);
            if (mode === "lego") {
              offCtx.fillStyle = "rgba(255, 255, 255, 0.4)";
              offCtx.beginPath();
              offCtx.arc(centerX, centerY, cellSize * 0.2, 0, Math.PI * 2);
              offCtx.fill();
            }
          }
        }
      }

      // ────────────────────────────────────────────────────────────────
      // PIPELINE STEP 4: Color Adjustments & Tints
      // ────────────────────────────────────────────────────────────────
      ctx.clearRect(0, 0, w, h);

      let filterCss = "";
      if (params.brightness !== 0) filterCss += ` brightness(${100 + params.brightness}%)`;
      if (params.contrast !== 100) filterCss += ` contrast(${params.contrast}%)`;
      if (params.saturation !== 100) filterCss += ` saturate(${params.saturation}%)`;
      if (params.grayscale > 0) filterCss += ` grayscale(${params.grayscale}%)`;

      ctx.filter = filterCss.trim() || "none";
      ctx.drawImage(offCanvas, 0, 0);
      ctx.filter = "none";

      // Tint Overlay
      if (params.tintOpacity > 0 && params.tint) {
        ctx.save();
        ctx.globalAlpha = params.tintOpacity / 100;
        ctx.globalCompositeOperation = params.overlayBlend || "multiply";
        ctx.fillStyle = params.tint;
        ctx.fillRect(0, 0, w, h);
        ctx.restore();
      }

      // ────────────────────────────────────────────────────────────────
      // PIPELINE STEP 5: Post-effects (pfx) Layering
      // ────────────────────────────────────────────────────────────────
      const pfx = params.pfx || {};

      // 5a. Bloom
      if (pfx.bloom && pfx.bloom.enabled) {
        ctx.save();
        ctx.globalCompositeOperation = "screen";
        ctx.globalAlpha = (pfx.bloom.intensity / 100) * 0.6;
        ctx.filter = "blur(14px)";
        ctx.drawImage(offCanvas, 0, 0);
        ctx.restore();
      }

      // 5b. Vignette
      if (pfx.vignette && pfx.vignette.enabled) {
        const vignInt = pfx.vignette.intensity / 100;
        const grad = ctx.createRadialGradient(
          w / 2, h / 2, Math.min(w, h) * 0.2,
          w / 2, h / 2, Math.max(w, h) * 0.75
        );
        grad.addColorStop(0, "rgba(0,0,0,0)");
        grad.addColorStop(1, `rgba(4,7,15,${0.85 * vignInt})`);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);
      }

      // 5c. Scan Lines
      if (pfx.scanLines && pfx.scanLines.enabled) {
        const scanInt = (pfx.scanLines.intensity / 100) * 0.3;
        ctx.fillStyle = `rgba(0, 0, 0, ${scanInt})`;
        for (let sy = 0; sy < h; sy += 4) {
          ctx.fillRect(0, sy, w, 1.5);
        }
      }

      // 5d. Chromatic Aberration
      if (pfx.chromatic && pfx.chromatic.enabled) {
        const shift = (pfx.chromatic.intensity / 100) * 4;
        ctx.save();
        ctx.globalCompositeOperation = "screen";
        ctx.globalAlpha = 0.3;
        ctx.drawImage(offCanvas, shift, 0);
        ctx.drawImage(offCanvas, -shift, 0);
        ctx.restore();
      }

      // 5e. Film Grain
      if (pfx.filmGrain && pfx.filmGrain.enabled) {
        const grainInt = (pfx.filmGrain.intensity / 100) * 0.15;
        const grainImg = ctx.createImageData(w, h);
        const gData = grainImg.data;
        for (let i = 0; i < gData.length; i += 4) {
          const val = (Math.random() - 0.5) * 255 * grainInt;
          gData[i] = val;
          gData[i + 1] = val;
          gData[i + 2] = val;
          gData[i + 3] = Math.abs(val);
        }
        const tempG = document.createElement('canvas');
        tempG.width = w; tempG.height = h;
        tempG.getContext('2d').putImageData(grainImg, 0, 0);
        ctx.drawImage(tempG, 0, 0);
      }

      // ────────────────────────────────────────────────────────────────
      // PIPELINE STEP 6: Light points glow
      // ────────────────────────────────────────────────────────────────
      if (params.lights && params.lights.enabled && params.lights.points) {
        params.lights.points.forEach(pt => {
          const lx = pt.x * w;
          const ly = pt.y * h;
          const lr = (pt.radius || 100);
          const lGrad = ctx.createRadialGradient(lx, ly, 0, lx, ly, lr);
          lGrad.addColorStop(0, `rgba(60, 166, 255, ${(pt.intensity || 50) / 100})`);
          lGrad.addColorStop(1, "rgba(60, 166, 255, 0)");
          ctx.fillStyle = lGrad;
          ctx.fillRect(0, 0, w, h);
        });
      }

      if (params.animated) {
        animFrameRef.current = requestAnimationFrame(renderFrame);
      }
    };

    animFrameRef.current = requestAnimationFrame(renderFrame);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [JSON.stringify(params), width, height]);

  return (
    <div className={`relative overflow-hidden rounded-2xl border border-sky-500/20 bg-slate-950 shadow-2xl ${className}`}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="block w-full h-full object-cover"
      />
    </div>
  );
}
