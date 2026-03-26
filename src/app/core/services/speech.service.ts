import { Injectable, inject, PLATFORM_ID } from '@angular/core';
import { BehaviorSubject, Observable, fromEvent, Subject } from 'rxjs';
import { takeUntil, debounceTime, filter, map } from 'rxjs/operators';
import { isPlatformBrowser } from '@angular/common';

export interface SpeechRecognitionResult {
  text: string;
  confidence: number;
  isFinal: boolean;
  timestamp: number;
}

@Injectable({
  providedIn: 'root'
})
export class SpeechService {
  private recognition: SpeechRecognition | null = null;
  private isRecording = false;
  private isInitialized = false;
  private isRecognizing = false; // <-- add this flag
  private destroy$ = new Subject<void>();
  
  private currentText = new BehaviorSubject<string>('');
  private currentConfidence = new BehaviorSubject<number>(0);
  private isRecordingSubject = new BehaviorSubject<boolean>(false);
  private errorSubject = new BehaviorSubject<string | null>(null);

  private platformId = inject(PLATFORM_ID);
  private isBrowser = isPlatformBrowser(this.platformId);
  private desiredLanguage: string = 'en-US';

  constructor() {
    if (this.isBrowser) {
      this.initializeSpeechRecognition();
    }
  }

  private initializeSpeechRecognition(): void {
    if (!this.isBrowser) {
      return;
    }
    const SpeechRecognition = (globalThis as any).SpeechRecognition || (globalThis as any).webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.error('Speech recognition not supported');
      this.errorSubject.next('browser-not-supported');
      return;
    }

    try {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.lang = this.desiredLanguage;
      this.recognition.maxAlternatives = 1;

      this.setupEventListeners();
      this.isInitialized = true;
      console.log('Speech service initialized successfully');
    } catch (error) {
      console.error('Failed to initialize speech recognition:', error);
      this.errorSubject.next('initialization-failed');
    }
  }

  private setupEventListeners(): void {
    if (!this.recognition) return;

    fromEvent(this.recognition, 'result')
      .pipe(takeUntil(this.destroy$))
      .subscribe((event: any) => {
        this.handleRecognitionResult(event);
      });

    fromEvent(this.recognition, 'error')
      .pipe(takeUntil(this.destroy$))
      .subscribe((event: any) => {
        this.handleRecognitionError(event);
      });

    fromEvent(this.recognition, 'start')
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => {
        console.log('Speech recognition started');
        this.isRecording = true;
        this.isRecognizing = true; // <-- set true on start
        this.isRecordingSubject.next(true);
        this.errorSubject.next(null);
      });

    fromEvent(this.recognition, 'end')
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => {
        console.log('Speech recognition ended');
        this.isRecording = false;
        this.isRecognizing = false; // <-- set false on end
        this.isRecordingSubject.next(false);
        // Emit empty string if nothing was captured
        if (!this.currentText.value.trim()) {
          this.currentText.next('');
        }
      });
  }

  private handleRecognitionResult(event: any): void {
    let finalTranscript = '';
    let interimTranscript = '';
    let maxConfidence = 0;

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      const transcript = result[0].transcript;
      const confidence = result[0].confidence || 0.8; // Default confidence if not provided
      
      maxConfidence = Math.max(maxConfidence, confidence);

      if (result.isFinal) {
        finalTranscript += transcript;
      } else {
        interimTranscript += transcript;
      }
    }

    const textToEmit = finalTranscript || interimTranscript;
    if (textToEmit.trim()) {
      console.log('Speech recognition result:', textToEmit, 'confidence:', maxConfidence);
      this.currentText.next(textToEmit);
      this.currentConfidence.next(maxConfidence);
    }
  }

  private handleRecognitionError(event: any): void {
    console.error('Speech recognition error:', event.error);
    
    if (event.error === 'not-allowed') {
      this.errorSubject.next('microphone-permission-denied');
    } else if (event.error === 'no-speech') {
      // Treat as transient; auto-restart to keep session alive
      this.errorSubject.next(null);
      setTimeout(() => {
        if (this.isRecording) {
          this.restart();
        }
      }, 300);
      return;
    } else if (event.error === 'audio-capture') {
      this.errorSubject.next('microphone-not-available');
    } else if (event.error === 'network') {
      this.errorSubject.next('network-error');
      setTimeout(() => {
        if (this.isRecording) {
          this.restart();
        }
      }, 1000);
      return;
    } else {
      this.errorSubject.next(event.error);
    }
  }

  async startRecording(): Promise<boolean> {
    if (!this.isBrowser) {
      console.error('Speech recognition not available on server');
      return false;
    }

    if (!this.isInitialized || !this.recognition) {
      this.initializeSpeechRecognition();
      if (!this.isInitialized || !this.recognition) {
        console.error('Speech recognition not initialized');
        return false;
      }
    }

    if (this.isRecording || this.isRecognizing) { // <-- check isRecognizing
      console.log('Already recording');
      return true;
    }

    try {
      // Request microphone permission first
      await this.requestMicrophonePermission();
      
      this.recognition.start();
      return true;
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      this.errorSubject.next('start-failed');
      return false;
    }
  }

  stopRecording(): void {
    if (!this.isRecording || !this.recognition) {
      return;
    }

    try {
      this.recognition.stop();
    } catch (error) {
      console.error('Error stopping speech recognition:', error);
    }
  }

  private restart(): void {
    if (!this.isRecording || !this.recognition) {
      return;
    }

    try {
      this.recognition.stop();
      setTimeout(() => {
        if (this.isRecording) {
          this.recognition!.start();
        }
      }, 100);
    } catch (error) {
      console.error('Error restarting speech recognition:', error);
    }
  }

  private async requestMicrophonePermission(): Promise<void> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      console.log('Microphone permission granted');
    } catch (error) {
      console.error('Microphone permission denied:', error);
      throw new Error('Microphone permission denied');
    }
  }

  setLanguage(lang: string): void {
    const normalized = this.normalizeLanguage(lang);
    this.desiredLanguage = normalized;
    if (this.recognition) {
      this.recognition.lang = normalized;
      console.log('Speech recognition language set to:', normalized);
    }
  }

  private normalizeLanguage(lang: string): string {
    if (!lang) return 'en-US';
    const lower = lang.toLowerCase();
    if (lower === 'en') return 'en-US';
    if (lower === 'hi') return 'hi-IN';
    if (lower === 'ta') return 'ta-IN';
    if (lower === 'te') return 'te-IN';
    if (lower === 'kn') return 'kn-IN';
    if (lower === 'ml') return 'ml-IN';
    if (lower === 'ur') return 'ur-IN';
    return lang;
  }

  // Public observables
  getCurrentText(): Observable<string> {
    return this.currentText.asObservable();
  }

  getCurrentConfidence(): Observable<number> {
    return this.currentConfidence.asObservable();
  }

  getIsRecording(): Observable<boolean> {
    return this.isRecordingSubject.asObservable();
  }

  getError(): Observable<string | null> {
    return this.errorSubject.asObservable();
  }

  getSpeechResult(): Observable<SpeechRecognitionResult> {
    return this.currentText.pipe(
      debounceTime(100),
      filter(text => text.trim().length > 0),
      map(text => ({
        text,
        confidence: 0.8, // Default confidence
        isFinal: true,
        timestamp: Date.now()
      })),
      takeUntil(this.destroy$)
    );
  }

  clearText(): void {
    this.currentText.next('');
    this.currentConfidence.next(0);
  }

  destroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.isRecording) {
      this.stopRecording();
    }
  }
}
