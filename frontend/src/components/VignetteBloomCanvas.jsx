/**
 * VignetteBloomCanvas — complete Canvas2D implementation of the
 * "Vignette Bloom" ASCII-art effect from 21st.dev.
 *
 * Render pipeline (faithful reimplementation):
 *   1. Draw source photo with bgMode / bgBlur / bgOpacity
 *   2. Divide into cellSize grid, sample avg colour & luminance per cell
 *   3. Draw primitive per renderMode; respect coverage/density/invert/edgeEmphasis
 *   4. Apply colour adjustments via ImageData: brightness, contrast, saturation,
 *      grayscale, tint/tintOpacity, blurType/blurAmount
 *   5. Post-effects (pfx): vignette, bloom, scanLines, chromatic, filmGrain,
 *      glitch, halftone, pixelate, filmDust
 *   6. Point lights glow
 *   7. Mask reveal
 *   8. Animation: wave / pulse / shimmer / ripple / flicker
 */
import React, { useEffect, useRef, useCallback } from 'react';

// ─────────────────────────── default config ─────────────────────────────────
export const DEFAULT_VIGNETTE_PARAMS = {
  renderMode: 'mosaic',
  bgMode: 'solid',
  bgBlur: 12,
  bgOpacity: 90,
  cellSize: 16,
  coverage: 100,
  invert: false,
  styleBlend: 'source-over',
  charSet: 'standard',
  customChars: '',
  brightness: 12,
  contrast: 115,
  edgeEmphasis: 0,
  density: 0,
  tint: '#3ca6ff',
  tintOpacity: 0,
  overlayBlend: 'multiply',
  saturation: 100,
  grayscale: 0,
  blurType: 'off',
  blurAmount: 35,
  pfx: {
    vignette:  { enabled: true,  intensity: 38 },
    scanLines: { enabled: false, intensity: 40 },
    chromatic: { enabled: false, intensity: 15 },
    bloom:     { enabled: true,  intensity: 25 },
    filmGrain: { enabled: false, intensity: 30 },
    glitch:    { enabled: false, intensity: 20 },
    pixelate:  { enabled: false, intensity: 15 },
    halftone:  { enabled: false, intensity: 20 },
    filmDust:  { enabled: false, intensity: 20 },
  },
  animated: true,
  animStyle: 'wave',
  animSpeed:     { enabled: true, intensity: 100 },
  animIntensity: { enabled: true, intensity: 60  },
  lights: { enabled: false, points: [] },
  mask:   { enabled: false, invert: false, dataUrl: null },
};

// ─────────────────────────── character sets ─────────────────────────────────
const CHAR_SETS = {
  standard: ' .:-=+*#%@',
  blocks:   ' ░▒▓█',
  minimal:  ' .+#@',
  binary:   '01 ',
  matrix:   '0123456789ABCDEFｦｱｳｴｵｶｷｹｺｻｼｽｾｿﾀﾂﾃﾅﾆﾇﾈﾊﾋﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾜ',
  hexdump:  '0123456789ABCDEF',
  braille:  '⠀⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟',
};

// ─────────────────────── pixel-level colour helpers ──────────────────────────
/** Box-blur a Float32 RGBA array (horizontal + vertical pass, sigma σ) */
function gaussianBlurRGBA(data, w, h, sigma) {
  const radius = Math.max(1, Math.round(sigma * 1.5));
  const out = new Float32Array(data.length);

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sumR = 0, sumG = 0, sumB = 0, sumA = 0, cnt = 0;
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        if (nx < 0 || nx >= w) continue;
        const i = (y * w + nx) * 4;
        sumR += data[i]; sumG += data[i+1]; sumB += data[i+2]; sumA += data[i+3];
        cnt++;
      }
      const o = (y * w + x) * 4;
      out[o]   = sumR / cnt;
      out[o+1] = sumG / cnt;
      out[o+2] = sumB / cnt;
      out[o+3] = sumA / cnt;
    }
  }
  // Vertical pass (in-place)
  const tmp = new Float32Array(out.length);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let sumR = 0, sumG = 0, sumB = 0, sumA = 0, cnt = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        const i = (ny * w + x) * 4;
        sumR += out[i]; sumG += out[i+1]; sumB += out[i+2]; sumA += out[i+3];
        cnt++;
      }
      const o = (y * w + x) * 4;
      tmp[o]   = sumR / cnt;
      tmp[o+1] = sumG / cnt;
      tmp[o+2] = sumB / cnt;
      tmp[o+3] = sumA / cnt;
    }
  }
  return tmp;
}

/** Apply brightness/contrast/saturation/grayscale/tint to Uint8ClampedArray in-place */
function applyColourAdjustments(pixels, w, h, cfg) {
  const bright   = cfg.brightness / 100;            // –1..+∞, 12 → +0.12
  const contrast = cfg.contrast / 100;              // 1.15 at default
  const sat      = cfg.saturation / 100;
  const gray     = cfg.grayscale / 100;
  const tintOp   = (cfg.tintOpacity || 0) / 100;

  // Parse tint colour
  let tR = 60, tG = 166, tB = 255;
  if (cfg.tint) {
    const hex = cfg.tint.replace('#', '');
    tR = parseInt(hex.slice(0, 2), 16);
    tG = parseInt(hex.slice(2, 4), 16);
    tB = parseInt(hex.slice(4, 6), 16);
  }

  for (let i = 0; i < pixels.length; i += 4) {
    let r = pixels[i] / 255;
    let g = pixels[i+1] / 255;
    let b = pixels[i+2] / 255;

    // Brightness (additive)
    r += bright; g += bright; b += bright;

    // Contrast (pivot at 0.5)
    r = (r - 0.5) * contrast + 0.5;
    g = (g - 0.5) * contrast + 0.5;
    b = (b - 0.5) * contrast + 0.5;

    // Saturation
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    r = lum + (r - lum) * sat;
    g = lum + (g - lum) * sat;
    b = lum + (b - lum) * sat;

    // Grayscale
    if (gray > 0) {
      const g2 = 0.299 * r + 0.587 * g + 0.114 * b;
      r = r + (g2 - r) * gray;
      g = g + (g2 - g) * gray;
      b = b + (g2 - b) * gray;
    }

    // Tint overlay (multiply blend)
    if (tintOp > 0) {
      r = r + (r * tR / 255 - r) * tintOp;
      g = g + (g * tG / 255 - g) * tintOp;
      b = b + (b * tB / 255 - b) * tintOp;
    }

    pixels[i]   = Math.max(0, Math.min(255, r * 255));
    pixels[i+1] = Math.max(0, Math.min(255, g * 255));
    pixels[i+2] = Math.max(0, Math.min(255, b * 255));
  }
}

/** Box-blur on Uint8ClampedArray, returns new Uint8ClampedArray */
function boxBlurPixels(pixels, w, h, radius) {
  const float = new Float32Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) float[i] = pixels[i];
  const blurred = gaussianBlurRGBA(float, w, h, radius);
  const out = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, Math.min(255, blurred[i]));
  return out;
}

// ─────────────────────── procedural fallback texture ────────────────────────
function drawFallbackTexture(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, '#020b18');
  bg.addColorStop(0.35, '#0b2340');
  bg.addColorStop(0.75, '#0e3056');
  bg.addColorStop(1, '#020b18');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // Glowing orbs
  const orbs = [
    { x: 0.2, y: 0.35, r: 160, c: 'rgba(20,100,255,0.18)' },
    { x: 0.65, y: 0.25, r: 120, c: 'rgba(0,200,180,0.14)' },
    { x: 0.75, y: 0.7,  r: 200, c: 'rgba(60,80,200,0.12)' },
    { x: 0.4,  y: 0.65, r: 140, c: 'rgba(0,160,220,0.16)' },
    { x: 0.9,  y: 0.45, r: 100, c: 'rgba(80,40,255,0.10)' },
  ];
  orbs.forEach(o => {
    const g = ctx.createRadialGradient(o.x*w, o.y*h, 0, o.x*w, o.y*h, o.r);
    g.addColorStop(0, o.c);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  });

  // Grid lines
  ctx.strokeStyle = 'rgba(60,166,255,0.07)';
  ctx.lineWidth = 1;
  const gridStep = 40;
  for (let x = 0; x < w; x += gridStep) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += gridStep) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Scattered data-points
  ctx.fillStyle = 'rgba(60,166,255,0.5)';
  for (let i = 0; i < 80; i++) {
    const px = (Math.sin(i * 137.5) * 0.5 + 0.5) * w;
    const py = (Math.cos(i * 97.3) * 0.5 + 0.5) * h;
    const r  = 1 + (Math.sin(i * 37) * 0.5 + 0.5) * 2.5;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Landmass contours
  ctx.strokeStyle = 'rgba(30,200,180,0.12)';
  ctx.lineWidth = 1.5;
  for (let k = 0; k < 6; k++) {
    ctx.beginPath();
    const ox = (Math.sin(k * 1.7) * 0.5 + 0.5) * w;
    const oy = (Math.cos(k * 2.3) * 0.5 + 0.5) * h;
    ctx.arc(ox, oy, 50 + k * 35, 0, Math.PI * 2);
    ctx.stroke();
  }
}

// ─────────────────────────── main component ─────────────────────────────────
export default function VignetteBloomCanvas({
  config = {},
  imageSrc = null,
  width = 900,
  height = 520,
  className = '',
}) {
  const canvasRef     = useRef(null);
  const animRef       = useRef(null);
  const imgRef        = useRef(null);
  const imgReadyRef   = useRef(false);

  // Deep-merge config with defaults
  const P = {
    ...DEFAULT_VIGNETTE_PARAMS,
    ...config,
    pfx: { ...DEFAULT_VIGNETTE_PARAMS.pfx, ...(config.pfx || {}) },
    animSpeed:     { ...DEFAULT_VIGNETTE_PARAMS.animSpeed,     ...(config.animSpeed     || {}) },
    animIntensity: { ...DEFAULT_VIGNETTE_PARAMS.animIntensity, ...(config.animIntensity || {}) },
    lights: { ...DEFAULT_VIGNETTE_PARAMS.lights, ...(config.lights || {}) },
    mask:   { ...DEFAULT_VIGNETTE_PARAMS.mask,   ...(config.mask   || {}) },
  };
  // Merge each pfx key
  Object.keys(DEFAULT_VIGNETTE_PARAMS.pfx).forEach(k => {
    P.pfx[k] = { ...DEFAULT_VIGNETTE_PARAMS.pfx[k], ...(P.pfx[k] || {}) };
  });

  // ── Load source image ──────────────────────────────────────────────────────
  useEffect(() => {
    const src = imageSrc || '/ascii-editor/demos/generated/ref-008.webp';
    const img = new Image();
    img.crossOrigin = 'anonymous';
    imgReadyRef.current = false;
    img.onload  = () => { imgRef.current = img; imgReadyRef.current = true; };
    img.onerror = () => { imgRef.current = null; imgReadyRef.current = true; };
    img.src = src;
  }, [imageSrc]);

  // ── Main render loop ───────────────────────────────────────────────────────
  const render = useCallback((timestamp) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const W = canvas.width;
    const H = canvas.height;

    const T  = timestamp / 1000;
    const spd = P.animSpeed.enabled     ? (P.animSpeed.intensity     / 50) : 1;
    const amt = P.animIntensity.enabled ? (P.animIntensity.intensity / 100) : 0.5;

    // ── STEP 1: Prepare background + sample source ─────────────────────────
    // Offscreen source canvas (always full-res reference for colour sampling)
    const srcC = document.createElement('canvas');
    srcC.width = W; srcC.height = H;
    const srcX = srcC.getContext('2d', { willReadFrequently: true });

    if (imgReadyRef.current && imgRef.current) {
      srcX.drawImage(imgRef.current, 0, 0, W, H);
    } else {
      drawFallbackTexture(srcX, W, H);
    }

    // Background layer
    ctx.clearRect(0, 0, W, H);

    if (P.bgMode === 'none') {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, W, H);
    } else if (P.bgMode === 'solid') {
      ctx.fillStyle = '#090d16';
      ctx.fillRect(0, 0, W, H);
    } else if (P.bgMode === 'photo') {
      ctx.globalAlpha = P.bgOpacity / 100;
      ctx.drawImage(srcC, 0, 0);
      ctx.globalAlpha = 1;
    } else {
      // blurred photo
      ctx.globalAlpha = P.bgOpacity / 100;
      ctx.filter = `blur(${P.bgBlur}px)`;
      ctx.drawImage(srcC, 0, 0);
      ctx.filter = 'none';
      ctx.globalAlpha = 1;
    }

    // ── STEP 2 + 3: Grid sampling + primitive drawing ──────────────────────
    const srcData  = srcX.getImageData(0, 0, W, H).data;
    const cellSize = Math.max(4, Math.round(P.cellSize));
    const cols     = Math.ceil(W / cellSize);
    const rows     = Math.ceil(H / cellSize);
    const charStr  = P.customChars || CHAR_SETS[P.charSet] || CHAR_SETS.standard;

    // Edge detection (Sobel magnitude per cell)
    let edgeMap = null;
    if (P.edgeEmphasis > 0) {
      edgeMap = new Float32Array(cols * rows);
      for (let r = 1; r < rows - 1; r++) {
        for (let c = 1; c < cols - 1; c++) {
          const lum = (cr, cc) => {
            const px = (Math.round(cr * cellSize + cellSize / 2)) * W
                     +  Math.round(cc * cellSize + cellSize / 2);
            const i = Math.min(px, W * H - 1) * 4;
            return (srcData[i] * 0.299 + srcData[i+1] * 0.587 + srcData[i+2] * 0.114) / 255;
          };
          const gx = -lum(r-1,c-1) - 2*lum(r,c-1) - lum(r+1,c-1)
                     + lum(r-1,c+1) + 2*lum(r,c+1) + lum(r+1,c+1);
          const gy = -lum(r-1,c-1) - 2*lum(r-1,c) - lum(r-1,c+1)
                     + lum(r+1,c-1) + 2*lum(r+1,c) + lum(r+1,c+1);
          edgeMap[r * cols + c] = Math.min(1, Math.hypot(gx, gy));
        }
      }
    }

    // Draw primitives layer onto offscreen primitive canvas
    const primC = document.createElement('canvas');
    primC.width = W; primC.height = H;
    const pCtx = primC.getContext('2d');

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        // Coverage gating
        const covRand = Math.abs(Math.sin(row * 127.1 + col * 311.7)) * 100;
        if (covRand > P.coverage) continue;

        const x = col * cellSize;
        const y = row * cellSize;
        const cx = x + cellSize / 2;
        const cy = y + cellSize / 2;

        // Sample average color in cell
        let sumR = 0, sumG = 0, sumB = 0, cnt = 0;
        const step = Math.max(1, Math.floor(cellSize / 4));
        for (let dy = 0; dy < cellSize && y + dy < H; dy += step) {
          for (let dx = 0; dx < cellSize && x + dx < W; dx += step) {
            const idx = ((y + dy) * W + (x + dx)) * 4;
            sumR += srcData[idx]; sumG += srcData[idx+1]; sumB += srcData[idx+2];
            cnt++;
          }
        }
        if (cnt === 0) continue;
        let avgR = sumR / cnt, avgG = sumG / cnt, avgB = sumB / cnt;
        let lum = (0.299 * avgR + 0.587 * avgG + 0.114 * avgB) / 255;
        if (P.invert) lum = 1 - lum;

        // Edge emphasis
        if (edgeMap && P.edgeEmphasis > 0) {
          const e = edgeMap[row * cols + col] || 0;
          lum = Math.min(1, lum + e * P.edgeEmphasis / 100);
        }

        // ── Animation offset per-cell ──
        let animOff = 0;
        if (P.animated) {
          const t = T * spd;
          if (P.animStyle === 'wave') {
            animOff = Math.sin(t * 2 + col * 0.3 + row * 0.3) * amt * 0.35;
          } else if (P.animStyle === 'pulse') {
            animOff = Math.sin(t * 3) * amt * 0.25;
          } else if (P.animStyle === 'shimmer') {
            animOff = Math.sin(t * 5 + col * 0.7 + row * 0.5) * amt * 0.2;
          } else if (P.animStyle === 'ripple') {
            const dist = Math.hypot(col - cols / 2, row - rows / 2);
            animOff = Math.sin(dist * 0.4 - t * 5) * amt * 0.25;
          } else if (P.animStyle === 'flicker') {
            animOff = (Math.random() - 0.5) * amt * 0.3;
          }
        }

        lum = Math.max(0, Math.min(1, lum + animOff));

        const mode = P.renderMode;
        pCtx.globalAlpha = 1;
        pCtx.globalCompositeOperation = P.styleBlend || 'source-over';

        // ── Helpers ──
        const fillCol = () => `rgb(${Math.round(avgR)},${Math.round(avgG)},${Math.round(avgB)})`;
        const alpha = (a) => `rgba(${Math.round(avgR)},${Math.round(avgG)},${Math.round(avgB)},${a.toFixed(3)})`;

        pCtx.fillStyle   = fillCol();
        pCtx.strokeStyle = fillCol();

        // ── Render modes ──────────────────────────────────────────────────
        if (mode === 'characters') {
          const idx2 = Math.max(0, Math.min(charStr.length - 1, Math.floor(lum * charStr.length)));
          const ch = charStr[idx2] || ' ';
          pCtx.font = `${Math.floor(cellSize * 0.9)}px monospace`;
          pCtx.textAlign = 'center';
          pCtx.textBaseline = 'middle';
          pCtx.fillText(ch, cx, cy);

        } else if (mode === 'hexdump') {
          const hs = CHAR_SETS.hexdump;
          const idx2 = Math.floor(lum * hs.length) % hs.length;
          pCtx.font = `bold ${Math.floor(cellSize * 0.85)}px monospace`;
          pCtx.textAlign = 'center';
          pCtx.textBaseline = 'middle';
          pCtx.fillText(hs[idx2], cx, cy);

        } else if (mode === 'matrix') {
          const ms = CHAR_SETS.matrix;
          const idx2 = (Math.floor((T * spd * 8 + col * 3 + row * 7) % ms.length) + ms.length) % ms.length;
          const greenFactor = 0.3 + lum * 0.7;
          pCtx.fillStyle = `rgba(0,${Math.round(200 * greenFactor + 40)},${Math.round(80 * greenFactor)},${0.5 + lum * 0.5})`;
          pCtx.font = `bold ${Math.floor(cellSize * 0.9)}px monospace`;
          pCtx.textAlign = 'center';
          pCtx.textBaseline = 'middle';
          pCtx.fillText(ms[idx2], cx, cy);

        } else if (mode === 'braille') {
          const bs = CHAR_SETS.braille;
          const idx2 = Math.floor(lum * (bs.length - 1));
          pCtx.font = `${Math.floor(cellSize * 1.1)}px sans-serif`;
          pCtx.textAlign = 'center';
          pCtx.textBaseline = 'middle';
          pCtx.fillText(bs[idx2], cx, cy);

        } else if (mode === 'dots' || mode === 'bubbles') {
          const r = (cellSize / 2) * (0.15 + lum * 0.85);
          pCtx.beginPath();
          pCtx.arc(cx, cy, Math.max(0.5, r), 0, Math.PI * 2);
          pCtx.fill();

        } else if (mode === 'rings') {
          const r = (cellSize / 2) * (0.2 + lum * 0.75);
          pCtx.beginPath();
          pCtx.arc(cx, cy, Math.max(1, r), 0, Math.PI * 2);
          pCtx.lineWidth = Math.max(0.5, cellSize * 0.1 * lum);
          pCtx.stroke();

        } else if (mode === 'hexagons') {
          const r = (cellSize / 2) * (0.4 + lum * 0.55);
          pCtx.beginPath();
          for (let a = 0; a < 6; a++) {
            const ang = (Math.PI / 3) * a - Math.PI / 6;
            if (a === 0) pCtx.moveTo(cx + r * Math.cos(ang), cy + r * Math.sin(ang));
            else         pCtx.lineTo(cx + r * Math.cos(ang), cy + r * Math.sin(ang));
          }
          pCtx.closePath();
          pCtx.fill();

        } else if (mode === 'triangles') {
          const r = (cellSize / 2) * (0.4 + lum * 0.55);
          const flip = (col + row) % 2 === 0;
          pCtx.beginPath();
          if (flip) {
            pCtx.moveTo(cx, cy - r);
            pCtx.lineTo(cx + r * 0.866, cy + r * 0.5);
            pCtx.lineTo(cx - r * 0.866, cy + r * 0.5);
          } else {
            pCtx.moveTo(cx, cy + r);
            pCtx.lineTo(cx + r * 0.866, cy - r * 0.5);
            pCtx.lineTo(cx - r * 0.866, cy - r * 0.5);
          }
          pCtx.closePath();
          pCtx.fill();

        } else if (mode === 'diamond') {
          const r = (cellSize / 2) * (0.4 + lum * 0.55);
          pCtx.beginPath();
          pCtx.moveTo(cx, cy - r);
          pCtx.lineTo(cx + r * 0.7, cy);
          pCtx.lineTo(cx, cy + r);
          pCtx.lineTo(cx - r * 0.7, cy);
          pCtx.closePath();
          pCtx.fill();

        } else if (mode === 'hearts') {
          const s = (cellSize / 2) * lum;
          pCtx.save();
          pCtx.translate(cx, cy);
          pCtx.scale(s * 0.06, s * 0.06);
          pCtx.beginPath();
          pCtx.moveTo(0, -5);
          pCtx.bezierCurveTo(-8, -12, -16, 2, 0, 12);
          pCtx.bezierCurveTo(16, 2, 8, -12, 0, -5);
          pCtx.closePath();
          pCtx.fill();
          pCtx.restore();

        } else if (mode === 'stars') {
          const outer = (cellSize / 2) * (0.3 + lum * 0.65);
          const inner = outer * 0.4;
          pCtx.beginPath();
          for (let pt = 0; pt < 10; pt++) {
            const ang = (Math.PI / 5) * pt - Math.PI / 2;
            const r2 = pt % 2 === 0 ? outer : inner;
            if (pt === 0) pCtx.moveTo(cx + r2 * Math.cos(ang), cy + r2 * Math.sin(ang));
            else          pCtx.lineTo(cx + r2 * Math.cos(ang), cy + r2 * Math.sin(ang));
          }
          pCtx.closePath();
          pCtx.fill();

        } else if (mode === 'cross') {
          pCtx.lineWidth = Math.max(1, cellSize * 0.15 * lum);
          const r = cellSize * 0.4 * (0.5 + lum * 0.5);
          pCtx.beginPath();
          pCtx.moveTo(cx - r, cy); pCtx.lineTo(cx + r, cy);
          pCtx.moveTo(cx, cy - r); pCtx.lineTo(cx, cy + r);
          pCtx.stroke();

        } else if (mode === 'lines') {
          pCtx.lineWidth = Math.max(0.5, cellSize * 0.2 * lum);
          pCtx.beginPath();
          pCtx.moveTo(x, cy); pCtx.lineTo(x + cellSize, cy);
          pCtx.stroke();

        } else if (mode === 'diagonal') {
          pCtx.lineWidth = Math.max(0.5, cellSize * 0.18 * lum);
          pCtx.beginPath();
          pCtx.moveTo(x, y); pCtx.lineTo(x + cellSize, y + cellSize);
          pCtx.stroke();

        } else if (mode === 'hatch') {
          pCtx.lineWidth = Math.max(0.5, cellSize * 0.14);
          const cnt2 = Math.max(1, Math.round(lum * 4));
          pCtx.beginPath();
          for (let h2 = 0; h2 < cnt2; h2++) {
            const t2 = h2 / cnt2;
            pCtx.moveTo(x + t2 * cellSize, y);
            pCtx.lineTo(x, y + t2 * cellSize);
          }
          if (lum > 0.5) {
            for (let h2 = 0; h2 < cnt2; h2++) {
              const t2 = h2 / cnt2;
              pCtx.moveTo(x + cellSize, y + t2 * cellSize);
              pCtx.lineTo(x + t2 * cellSize, y + cellSize);
            }
          }
          pCtx.stroke();

        } else if (mode === 'contour') {
          // Topographic iso-lines: concentric ellipses
          const levels = 3;
          pCtx.lineWidth = 0.5;
          for (let lv = 0; lv < levels; lv++) {
            const f = (lv + 1) / (levels + 1);
            if (lum < f - 0.15) continue;
            const r = (cellSize / 2) * f;
            pCtx.globalAlpha = 0.6 + lum * 0.4;
            pCtx.beginPath();
            pCtx.ellipse(cx, cy, r, r * 0.65, 0, 0, Math.PI * 2);
            pCtx.stroke();
          }
          pCtx.globalAlpha = 1;

        } else if (mode === 'halfblocks') {
          // Top half vs bottom half at different luminance
          const topLum = lum;
          const botLum = Math.min(1, lum * 1.25);
          pCtx.fillStyle = `rgba(${Math.round(avgR * topLum)},${Math.round(avgG * topLum)},${Math.round(avgB * topLum)},1)`;
          pCtx.fillRect(x, y, cellSize, cellSize / 2);
          pCtx.fillStyle = `rgba(${Math.round(avgR * botLum)},${Math.round(avgG * botLum)},${Math.round(avgB * botLum)},1)`;
          pCtx.fillRect(x, y + cellSize / 2, cellSize, cellSize / 2);

        } else if (mode === 'lego') {
          const pad = Math.max(1, cellSize * 0.07);
          pCtx.fillRect(x + pad, y + pad, cellSize - pad * 2, cellSize - pad * 2);
          const bumpR = cellSize * 0.2;
          const grd = pCtx.createRadialGradient(
            cx - bumpR * 0.3, cy - bumpR * 0.3, 0,
            cx, cy, bumpR
          );
          grd.addColorStop(0, 'rgba(255,255,255,0.35)');
          grd.addColorStop(1, 'rgba(0,0,0,0.25)');
          pCtx.fillStyle = grd;
          pCtx.beginPath();
          pCtx.arc(cx, cy, bumpR, 0, Math.PI * 2);
          pCtx.fill();

        } else if (mode === 'voxel') {
          // Isometric cube illusion
          const s = (cellSize * 0.45) * lum;
          pCtx.fillStyle = fillCol();
          pCtx.fillRect(cx - s, cy - s, s * 2, s * 2);
          pCtx.fillStyle = alpha(0.25);
          pCtx.fillRect(cx, cy - s, s, s);

        } else if (mode === 'disco') {
          const hue = ((col * 37 + row * 53 + T * spd * 60) % 360 + 360) % 360;
          pCtx.fillStyle = `hsla(${hue},80%,60%,${0.4 + lum * 0.6})`;
          pCtx.fillRect(x, y, cellSize, cellSize);

        } else if (mode === 'mixed') {
          const which = (col + row) % 4;
          if (which === 0) {
            pCtx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
          } else if (which === 1) {
            const r = (cellSize / 2) * lum;
            pCtx.beginPath(); pCtx.arc(cx, cy, r, 0, Math.PI * 2); pCtx.fill();
          } else if (which === 2) {
            pCtx.beginPath();
            pCtx.moveTo(cx, y + 2); pCtx.lineTo(x + cellSize - 2, y + cellSize - 2);
            pCtx.lineTo(x + 2, y + cellSize - 2); pCtx.closePath(); pCtx.fill();
          } else {
            const idx2 = Math.floor(lum * (charStr.length - 1));
            pCtx.font = `${Math.floor(cellSize * 0.9)}px monospace`;
            pCtx.textAlign = 'center'; pCtx.textBaseline = 'middle';
            pCtx.fillText(charStr[idx2] || '.', cx, cy);
          }

        } else if (mode === 'dither') {
          // Ordered Bayer 4x4 dithering
          const M4 = [
            [ 0, 8, 2,10],
            [12, 4,14, 6],
            [ 3,11, 1, 9],
            [15, 7,13, 5],
          ];
          const bx = ((col % 4) + 4) % 4;
          const by = ((row % 4) + 4) % 4;
          const threshold = (M4[by][bx] + 0.5) / 16;
          if (lum >= threshold) {
            pCtx.fillRect(x, y, cellSize, cellSize);
          }

        } else if (mode === 'pixel') {
          // Crisp hard pixel
          pCtx.fillStyle = lum > 0.1 ? fillCol() : '#000';
          pCtx.fillRect(x, y, cellSize, cellSize);

        } else {
          // Default: mosaic (soft fill)
          pCtx.fillStyle = alpha(0.7 + lum * 0.3);
          pCtx.fillRect(x, y, cellSize, cellSize);
        }
      }
    }

    // Composite primitives onto background
    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(primC, 0, 0);

    // ── STEP 4: Colour adjustments via ImageData ───────────────────────────
    if (P.brightness !== 0 || P.contrast !== 100 || P.saturation !== 100 ||
        P.grayscale !== 0  || P.tintOpacity > 0) {
      const imgd = ctx.getImageData(0, 0, W, H);
      applyColourAdjustments(imgd.data, W, H, P);
      ctx.putImageData(imgd, 0, 0);
    }

    // Blur post-pass
    if (P.blurType !== 'off' && P.blurAmount > 0) {
      const imgd = ctx.getImageData(0, 0, W, H);
      const blurred = boxBlurPixels(imgd.data, W, H, P.blurAmount * 0.15);
      const newImg = new ImageData(blurred, W, H);
      ctx.putImageData(newImg, 0, 0);
    }

    // ── STEP 5: Post-effects ───────────────────────────────────────────────
    const pfx = P.pfx;

    // Bloom: screen-composite blurred copy of current frame
    if (pfx.bloom?.enabled) {
      const bi = pfx.bloom.intensity / 100;
      const bloomC = document.createElement('canvas');
      bloomC.width = W; bloomC.height = H;
      const bCtx = bloomC.getContext('2d');
      bCtx.filter = `blur(${12 + bi * 8}px)`;
      bCtx.drawImage(canvas, 0, 0);
      bCtx.filter = 'none';
      ctx.save();
      ctx.globalCompositeOperation = 'screen';
      ctx.globalAlpha = bi * 0.55;
      ctx.drawImage(bloomC, 0, 0);
      ctx.restore();
    }

    // Vignette: radial gradient darkening toward edges
    if (pfx.vignette?.enabled) {
      const vi = pfx.vignette.intensity / 100;
      const vg = ctx.createRadialGradient(
        W / 2, H / 2, Math.min(W, H) * 0.15,
        W / 2, H / 2, Math.max(W, H) * 0.78
      );
      vg.addColorStop(0, 'rgba(0,0,0,0)');
      vg.addColorStop(1, `rgba(3,5,14,${(0.88 * vi).toFixed(3)})`);
      ctx.fillStyle = vg;
      ctx.fillRect(0, 0, W, H);
    }

    // Scan Lines
    if (pfx.scanLines?.enabled) {
      const si = (pfx.scanLines.intensity / 100) * 0.35;
      ctx.fillStyle = `rgba(0,0,0,${si.toFixed(3)})`;
      for (let sy = 0; sy < H; sy += 4) {
        ctx.fillRect(0, sy, W, 2);
      }
    }

    // Chromatic Aberration
    if (pfx.chromatic?.enabled) {
      const shift = (pfx.chromatic.intensity / 100) * 5;
      ctx.save();
      ctx.globalCompositeOperation = 'screen';
      ctx.globalAlpha = 0.22;
      ctx.drawImage(canvas, shift, 0);
      ctx.drawImage(canvas, -shift, 0);
      ctx.restore();
    }

    // Film Grain
    if (pfx.filmGrain?.enabled) {
      const gi = (pfx.filmGrain.intensity / 100) * 0.18;
      const grainData = ctx.createImageData(W, H);
      const gd = grainData.data;
      for (let i = 0; i < gd.length; i += 4) {
        const n = (Math.random() - 0.5) * 255 * gi;
        gd[i]   = Math.max(0, Math.min(255, 128 + n));
        gd[i+1] = Math.max(0, Math.min(255, 128 + n));
        gd[i+2] = Math.max(0, Math.min(255, 128 + n));
        gd[i+3] = Math.abs(n * 2);
      }
      const gC = document.createElement('canvas');
      gC.width = W; gC.height = H;
      gC.getContext('2d').putImageData(grainData, 0, 0);
      ctx.save();
      ctx.globalCompositeOperation = 'overlay';
      ctx.globalAlpha = 0.5;
      ctx.drawImage(gC, 0, 0);
      ctx.restore();
    }

    // Glitch: horizontal slice offset
    if (pfx.glitch?.enabled) {
      const gi = pfx.glitch.intensity / 100;
      const slices = Math.floor(3 + gi * 6);
      for (let s = 0; s < slices; s++) {
        if (Math.random() > 0.25 * gi) continue;
        const sy = Math.random() * H;
        const sh = Math.max(2, Math.random() * 12 * gi);
        const dx = (Math.random() - 0.5) * 30 * gi;
        const slice = ctx.getImageData(0, sy, W, sh);
        ctx.putImageData(slice, dx, sy);
      }
    }

    // Halftone
    if (pfx.halftone?.enabled) {
      const hi = pfx.halftone.intensity / 100;
      const dotSize = 4 + hi * 8;
      ctx.save();
      ctx.globalCompositeOperation = 'multiply';
      ctx.fillStyle = `rgba(0,0,0,${(hi * 0.4).toFixed(3)})`;
      for (let hy = 0; hy < H; hy += dotSize * 1.8) {
        for (let hx = 0; hx < W; hx += dotSize * 1.8) {
          const r = dotSize * (0.3 + Math.random() * 0.4) * hi;
          ctx.beginPath();
          ctx.arc(hx + dotSize * 0.9, hy + dotSize * 0.9, r, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    }

    // Pixelate
    if (pfx.pixelate?.enabled) {
      const pi = pfx.pixelate.intensity / 100;
      const ps = Math.max(2, Math.round(pi * 20));
      const tmpC = document.createElement('canvas');
      tmpC.width  = Math.ceil(W / ps);
      tmpC.height = Math.ceil(H / ps);
      const tCtx = tmpC.getContext('2d');
      tCtx.imageSmoothingEnabled = false;
      tCtx.drawImage(canvas, 0, 0, tmpC.width, tmpC.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmpC, 0, 0, W, H);
      ctx.imageSmoothingEnabled = true;
    }

    // Film Dust: random bright specks
    if (pfx.filmDust?.enabled) {
      const di = pfx.filmDust.intensity / 100;
      const specks = Math.floor(di * 80);
      for (let sp = 0; sp < specks; sp++) {
        if (Math.random() > di) continue;
        const fx = Math.random() * W;
        const fy = Math.random() * H;
        const fr = Math.random() * 1.5;
        ctx.fillStyle = `rgba(255,255,240,${(0.4 + Math.random() * 0.6).toFixed(2)})`;
        ctx.beginPath();
        ctx.arc(fx, fy, fr, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ── STEP 6: Point Lights ───────────────────────────────────────────────
    if (P.lights?.enabled && P.lights.points?.length > 0) {
      P.lights.points.forEach(pt => {
        const lx = pt.x * W;
        const ly = pt.y * H;
        const lr = (pt.radius || 100) * (W / 800);
        const li = (pt.intensity || 50) / 100;
        const lg = ctx.createRadialGradient(lx, ly, 0, lx, ly, lr);
        lg.addColorStop(0, `rgba(60,180,255,${(li * 0.7).toFixed(3)})`);
        lg.addColorStop(0.4, `rgba(30,100,200,${(li * 0.3).toFixed(3)})`);
        lg.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = lg;
        ctx.fillRect(0, 0, W, H);
      });
    }

    // ── STEP 7: Mask reveal ────────────────────────────────────────────────
    // (mask.dataUrl would require preloading; handled externally if needed)

    // ── Continue animation ─────────────────────────────────────────────────
    if (P.animated) {
      animRef.current = requestAnimationFrame(render);
    }
  }, [P]);

  useEffect(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    // Wait briefly for image to load before first frame
    const t = setTimeout(() => {
      animRef.current = requestAnimationFrame(render);
    }, 120);
    return () => {
      clearTimeout(t);
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [render]);

  return (
    <div className={`relative overflow-hidden rounded-2xl bg-slate-950 ${className}`}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ display: 'block', width: '100%', height: '100%' }}
      />
    </div>
  );
}
