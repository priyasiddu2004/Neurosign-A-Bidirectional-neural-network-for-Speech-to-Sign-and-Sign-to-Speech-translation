import {Injectable, PLATFORM_ID, inject} from '@angular/core';
import {BehaviorSubject, Observable} from 'rxjs';
import {isPlatformBrowser} from '@angular/common';

@Injectable({providedIn: 'root'})
export class OnDeviceSttService {
  private platformId = inject(PLATFORM_ID);
  private isBrowser = isPlatformBrowser(this.platformId);

  private vosk: any = null;
  private recognizer: any = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;

  private ready$ = new BehaviorSubject<boolean>(false);
  private text$ = new BehaviorSubject<string>('');
  private isRecording$ = new BehaviorSubject<boolean>(false);
  private error$ = new BehaviorSubject<string | null>(null);

  /**
   * Lazily load Vosk WASM and model. Call once before start.
   * modelPath should point to a folder containing model files (e.g., src/assets/stt/vosk/en-us/).
   */
  async init(modelPath: string): Promise<boolean> {
    if (!this.isBrowser) {
      return false;
    }
    try {
      if (!this.vosk) {
        this.vosk = await import('vosk-browser');
      }

      if (this.vosk && (this.vosk as any).setWasmPrefix) {
        (this.vosk as any).setWasmPrefix('');
      }

      await this.vosk.init();

      const model = await this.vosk.createModel(modelPath);
      this.recognizer = await model.createRecognizer(16000);
      this.ready$.next(true);
      return true;
    } catch (e) {
      console.error('OnDevice STT init failed:', e);
      this.error$.next('on-device-init-failed');
      return false;
    }
  }

  async start(): Promise<boolean> {
    if (!this.isBrowser) {
      return false;
    }
    if (!this.recognizer) {
      this.error$.next('on-device-not-ready');
      return false;
    }
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});
      this.source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

      this.processor.onaudioprocess = (event: AudioProcessingEvent) => {
        const input = event.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          let s = Math.max(-1, Math.min(1, input[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        const result = this.recognizer.acceptWaveform(pcm16);
        if (result) {
          const res = this.recognizer.result();
          if (res && res.text) {
            this.text$.next(res.text);
          }
        } else {
          const res = this.recognizer.partialResult();
          if (res && res.partial) {
            this.text$.next(res.partial);
          }
        }
      };

      this.source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);
      this.isRecording$.next(true);
      return true;
    } catch (e) {
      console.error('OnDevice STT start failed:', e);
      this.error$.next('on-device-start-failed');
      return false;
    }
  }

  stop(): void {
    try {
      this.isRecording$.next(false);
      if (this.processor) {
        this.processor.disconnect();
        this.processor.onaudioprocess = null as any;
        this.processor = null;
      }
      if (this.source) {
        this.source.disconnect();
        this.source = null;
      }
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach(t => t.stop());
        this.mediaStream = null;
      }
      if (this.audioContext) {
        this.audioContext.close();
        this.audioContext = null;
      }
    } catch (e) {
      console.error('OnDevice STT stop failed:', e);
    }
  }

  destroy(): void {
    this.stop();
    this.ready$.complete();
    this.text$.complete();
    this.isRecording$.complete();
    this.error$.complete();
  }

  getReady(): Observable<boolean> {
    return this.ready$.asObservable();
  }

  getText(): Observable<string> {
    return this.text$.asObservable();
  }

  getIsRecording(): Observable<boolean> {
    return this.isRecording$.asObservable();
  }

  getError(): Observable<string | null> {
    return this.error$.asObservable();
  }
}


