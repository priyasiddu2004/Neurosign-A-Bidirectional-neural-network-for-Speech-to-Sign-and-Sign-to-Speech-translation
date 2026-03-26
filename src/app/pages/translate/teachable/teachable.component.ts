import {Component, NgZone} from '@angular/core';
import {IonButton} from '@ionic/angular/standalone';

@Component({
  selector: 'app-teachable',
  standalone: true,
  templateUrl: './teachable.component.html',
  styleUrls: ['./teachable.component.scss'],
  imports: [IonButton]
})
export class TeachableComponent {
  tmLabel = '';
  tmConfidence = 0;
  accumulated = '';
  tmStatus = '';
  tmError = '';
  private model: any = null;
  private webcam: any = null;
  private raf: number | null = null;
  private running = false;
  // 3-second capture window tracking
  private windowStart = 0;
  private bestInWindowLabel = '';
  private bestInWindowProb = 0;

  constructor(private zone: NgZone) {}

  async start(): Promise<void> {
    if (this.running) return; // guard against double start
    if (!(window as any).tf) {
      await this.loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js');
    }
    if (!(window as any).tmPose) {
      await this.loadScript('https://cdn.jsdelivr.net/npm/@teachablemachine/pose@0.8/dist/teachablemachine-pose.min.js');
    }
    const tmPose = (window as any).tmPose;
    if (!tmPose) { this.tmError = 'tmPose not available'; return; }
    const base = 'assets/models/sign models/'.replace(/\s/g, '%20');
    const modelURL = `${base}model.json`;
    const metadataURL = `${base}metadata.json`;
    this.tmStatus = 'Loading model...'; this.tmError = '';
    try { this.model = await tmPose.load(modelURL, metadataURL); }
    catch { this.tmError = 'Model load failed'; this.tmStatus = ''; return; }

    const size = 300; const flip = true;
    this.webcam = new tmPose.Webcam(size, size, flip);
    try { await this.webcam.setup({ facingMode: 'user' }); } catch {}
    if (!this.webcam.stream) { try { await this.webcam.setup({ facingMode: 'environment' }); } catch {} }
    if (!this.webcam.stream) { try { await this.webcam.setup(); } catch { this.tmError = 'Camera setup failed'; return; } }
    try { await this.webcam.play(); } catch { this.tmError = 'Camera play failed'; return; }
    this.running = true;
    this.windowStart = performance.now();
    this.bestInWindowLabel = '';
    this.bestInWindowProb = 0;
    this.tmStatus = 'Running';

    const canvas = document.getElementById('tm-standalone') as HTMLCanvasElement | null;
    if (!canvas) return;
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext('2d');

    const loop = async () => {
      if (!this.running) return;
      this.webcam.update();
      const pose = await this.model.estimatePose(this.webcam.canvas);
      const prediction = await this.model.predict(pose.posenetOutput);
      let best = { className: '', probability: 0 } as any;
      for (const p of prediction) if (p.probability > best.probability) best = p;
      this.zone.run(() => {
        this.tmLabel = best.className;
        this.tmConfidence = best.probability || 0;
        // Track best within the 3-second window
        if ((best.probability || 0) > this.bestInWindowProb) {
          this.bestInWindowProb = best.probability || 0;
          this.bestInWindowLabel = best.className;
        }
      });
      if (ctx) {
        ctx.drawImage(this.webcam.canvas, 0, 0);
        (window as any).tmPose.drawKeypoints(pose.pose.keypoints, 0.5, ctx);
        (window as any).tmPose.drawSkeleton(pose.pose.keypoints, 0.5, ctx);
      }
      // Every 3 seconds, capture the best token once
      const now = performance.now();
      if (now - this.windowStart >= 3000) {
        this.zone.run(() => this.updateAccumulation(this.bestInWindowLabel, this.bestInWindowProb));
        this.windowStart = now;
        this.bestInWindowLabel = '';
        this.bestInWindowProb = 0;
      }

      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop(): void {
    if (this.raf != null) cancelAnimationFrame(this.raf);
    try { this.webcam?.stop?.(); } catch {}
    this.raf = null;
    this.running = false;
  }

  private loadScript(src: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src; s.async = true; s.defer = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('Failed to load ' + src));
      document.body.appendChild(s);
    });
  }

  // ----------- Accumulation with stability and pause-to-space -----------
  private lastCandidate: string | null = null;
  private stableCount = 0;
  private lastSeenTs = 0;
  private readonly requiredStableFrames = 3;
  private readonly pauseMsForSpace = 800;
  private readonly minConfidence = 0.5;

  private updateAccumulation(label: string, prob: number): void {
    const token = (label || '').trim();
    const now = Date.now();

    if (token && prob >= this.minConfidence && token.toUpperCase() !== 'UNKNOWN') {
      if (this.lastCandidate === token) {
        this.stableCount++;
      } else {
        this.lastCandidate = token;
        this.stableCount = 1;
      }
      this.lastSeenTs = now;

      if (this.stableCount >= this.requiredStableFrames) {
        // Append token and a trailing space if last char isn't space
        this.accumulated += (this.accumulated && !this.accumulated.endsWith(' ')) ? ' ' + token : token;
        this.stableCount = 0;
        this.lastCandidate = null;
      }
    } else {
      // No confident token; if paused long enough add a space once
      if (this.lastSeenTs > 0 && now - this.lastSeenTs > this.pauseMsForSpace) {
        if (!this.accumulated.endsWith(' ') && this.accumulated.length > 0) {
          this.accumulated += ' ';
        }
        this.lastSeenTs = 0;
        this.stableCount = 0;
        this.lastCandidate = null;
      }
    }
  }

  clearAccumulated(): void { this.accumulated = ''; }
}


