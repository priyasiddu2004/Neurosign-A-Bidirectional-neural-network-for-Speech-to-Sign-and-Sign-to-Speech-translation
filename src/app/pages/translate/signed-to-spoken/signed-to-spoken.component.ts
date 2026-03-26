import {Component, inject, OnInit, OnDestroy} from '@angular/core';
import {Router} from '@angular/router';
import {Store} from '@ngxs/store';
import {VideoStateModel} from '../../../core/modules/ngxs/store/video/video.state';
import {InputMode} from '../../../modules/translate/translate.state';
import {
  CopySpokenLanguageText,
  SetSignWritingText,
  SetSpokenLanguageText,
} from '../../../modules/translate/translate.actions';
import {Observable, Subject} from 'rxjs';
import {takeUntil, tap, debounceTime, distinctUntilChanged} from 'rxjs/operators';
import {MatTooltipModule} from '@angular/material/tooltip';
import {SignWritingComponent} from '../signwriting/sign-writing.component';
import {IonButton, IonIcon} from '@ionic/angular/standalone';
import {TextToSpeechComponent} from '../../../components/text-to-speech/text-to-speech.component';
import {UploadComponent} from './upload/upload.component';
import {addIcons} from 'ionicons';
import {copyOutline, school} from 'ionicons/icons';
import {TranslocoPipe} from '@jsverse/transloco';
import {AsyncPipe, NgTemplateOutlet} from '@angular/common';
import {VideoModule} from '../../../components/video/video.module';
import {SignToTextTranslationService, SignToTextResult} from '../../../modules/translate/sign-to-text-translation.service';

// Removed FAKE_WORDS - now using real sign detection

@Component({
  selector: 'app-signed-to-spoken',
  templateUrl: './signed-to-spoken.component.html',
  styleUrls: ['./signed-to-spoken.component.scss'],
  imports: [
    MatTooltipModule,
    SignWritingComponent,
    IonButton,
    TextToSpeechComponent,
    VideoModule,
    UploadComponent,
    IonIcon,
    TranslocoPipe,
    AsyncPipe,
    NgTemplateOutlet,
  ],
})
export class SignedToSpokenComponent implements OnInit, OnDestroy {
  private store = inject(Store);
  private router = inject(Router);
  private signToTextService = inject(SignToTextTranslationService);
  private destroy$ = new Subject<void>();

  videoState$!: Observable<VideoStateModel>;
  inputMode$!: Observable<InputMode>;
  spokenLanguage$!: Observable<string>;
  spokenLanguageText$!: Observable<string>;
  
  // Real-time sign detection properties
  currentSignText$!: Observable<string>;
  isSigning$!: Observable<boolean>;
  confidence$!: Observable<number>;

  // Teachable Machine (standalone) fields
  tmLabel = '';
  tmRunning = false;
  private tmModel: any = null;
  private tmWebcam: any = null;
  private tmRaf: number | null = null;

  constructor() {
    this.videoState$ = this.store.select<VideoStateModel>(state => state.video);
    this.inputMode$ = this.store.select<InputMode>(state => state.translate.inputMode);
    this.spokenLanguage$ = this.store.select<string>(state => state.translate.spokenLanguage);
    this.spokenLanguageText$ = this.store.select<string>(state => state.translate.spokenLanguageText);

    // Initialize sign detection observables
    this.currentSignText$ = this.signToTextService.getCurrentToken();
    this.isSigning$ = this.signToTextService.getIsSigning();
    this.confidence$ = this.signToTextService.getCurrentConfidence();

    this.store.dispatch(new SetSpokenLanguageText(''));

    addIcons({copyOutline, school});
  }

  ngOnInit(): void {
    console.log('SignedToSpokenComponent initialized');
    console.log('SignToTextService:', this.signToTextService);

    // Initialize with empty text
    this.store.dispatch(new SetSpokenLanguageText(''));

    // Subscribe to real-time sign detection
    this.signToTextService.getSignToTextResult()
      .pipe(
        debounceTime(120), // Faster updates for continuous live detection
        distinctUntilChanged((prev, curr) => prev.text === curr.text),
        tap((result: SignToTextResult) => {
          console.log('Sign detection result:', result);
          if (result.text) { // Removed confidence threshold for now
            // Update the spoken language text with detected sign
            console.log('Dispatching new text:', result.text);
            this.store.dispatch(new SetSpokenLanguageText(result.text));
          }
        }),
        takeUntil(this.destroy$)
      )
      .subscribe();

    // Also listen to individual text changes for debugging
    this.currentSignText$
      .pipe(
        tap(text => {
          console.log('Detected sign text:', text);
        }),
        takeUntil(this.destroy$)
      )
      .subscribe();

    // Listen to confidence changes for debugging
    this.confidence$
      .pipe(
        tap(confidence => {
          console.log('Sign detection confidence:', confidence);
        }),
        takeUntil(this.destroy$)
      )
      .subscribe();

    // Listen to signing status changes
    this.isSigning$
      .pipe(
        tap(isSigning => {
          console.log('Is signing:', isSigning);
        }),
        takeUntil(this.destroy$)
      )
      .subscribe();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();

    // Stop TM loop if running
    if (this.tmRaf != null) {
      cancelAnimationFrame(this.tmRaf);
      this.tmRaf = null;
    }
    try { this.tmWebcam?.stop?.(); } catch {}
  }

  copyTranslation() {
    this.store.dispatch(CopySpokenLanguageText);
  }

  navigateToTeachable(): void {
    this.router.navigate(['/teachable']);
  }

  // =========== Teachable Machine (local model) ===========
  async startTeachableMachine(): Promise<void> {
    // Load scripts dynamically (browser only)
    if (!(window as any).tf) {
      await this.loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js');
    }
    if (!(window as any).tmPose) {
      await this.loadScript('https://cdn.jsdelivr.net/npm/@teachablemachine/pose@0.8/dist/teachablemachine-pose.min.js');
    }
    const tmPose = (window as any).tmPose;
    if (!tmPose) { return; }

    // Load model from assets/models/sign models/
    const base = 'assets/models/sign models/'.replace(/\s/g, '%20');
    const modelURL = `${base}model.json`;
    const metadataURL = `${base}metadata.json`;
    this.tmModel = await tmPose.load(modelURL, metadataURL);

    // Setup webcam on our canvas
    const size = 224; const flip = true;
    this.tmWebcam = new tmPose.Webcam(size, size, flip);
    await this.tmWebcam.setup();
    await this.tmWebcam.play();
    this.tmRunning = true;

    // Attach to canvas
    const canvas = document.getElementById('tm-canvas') as HTMLCanvasElement | null;
    if (!canvas) { return; }
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext('2d');

    const loop = async () => {
      this.tmWebcam.update();
      const { posenetOutput } = await this.tmModel.estimatePose(this.tmWebcam.canvas);
      const prediction = await this.tmModel.predict(posenetOutput);
      let best = { className: '', probability: 0 } as any;
      for (const p of prediction) if (p.probability > best.probability) best = p;
      this.tmLabel = best.className;
      if (ctx) {
        ctx.drawImage(this.tmWebcam.canvas, 0, 0);
        if ((window as any).tmPose) {
          const pose = await this.tmModel.estimatePose(this.tmWebcam.canvas);
          (window as any).tmPose.drawKeypoints(pose.pose.keypoints, 0.5, ctx);
          (window as any).tmPose.drawSkeleton(pose.pose.keypoints, 0.5, ctx);
        }
      }
      if (this.tmRunning) {
        this.tmRaf = requestAnimationFrame(loop);
      }
    };
    this.tmRaf = requestAnimationFrame(loop);
  }

  stopTeachableMachine(): void {
    this.tmRunning = false;
    if (this.tmRaf != null) { cancelAnimationFrame(this.tmRaf); this.tmRaf = null; }
    try { this.tmWebcam?.stop?.(); } catch {}
  }

  private loadScript(src: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.async = true;
      s.defer = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('Failed to load ' + src));
      document.body.appendChild(s);
    });
  }
}
