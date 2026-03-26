import {inject, Injectable} from '@angular/core';
import {Observable, BehaviorSubject, combineLatest} from 'rxjs';
import {map, filter, debounceTime, distinctUntilChanged, tap} from 'rxjs/operators';
import {Store} from '@ngxs/store';
import {EstimatedPose} from '../pose/pose.state';
import {DetectorStateModel} from '../detector/detector.state';
import {PoseService} from '../pose/pose.service';
import {TensorflowService} from '../../core/services/tfjs/tfjs.service';
// Fingerpose is optional; we import dynamically at runtime in browser

export interface SignToTextResult {
  text: string;
  confidence: number;
  timestamp: number;
}

@Injectable({
  providedIn: 'root',
})
export class SignToTextTranslationService {
  private store = inject(Store);
  private poseService = inject(PoseService);
  private tfService = inject(TensorflowService);

  private currentText = new BehaviorSubject<string>('');
  private currentToken = new BehaviorSubject<string>('');
  private currentConfidence = new BehaviorSubject<number>(0);
  private isSigning = new BehaviorSubject<boolean>(false);
  private detectedSigns: string[] = []; // accepted tokens (letters/digits/spaces)

  // Stability and spacing controls for reliable live transcription
  private lastCandidate: string | null = null;
  private lastAccepted: string | null = null;
  private stableCount = 0;
  private readonly requiredStableFrames = 3; // frames that must agree before accepting
  private lastSeenTs = 0;
  private readonly pauseMsForSpace = 700; // add space when idle for this long
  private fingerpose: any | null = null;
  private gestureEstimator: any | null = null;
  private alphabetModel: any | null = null;
  private alphabetLabels: string[] = Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ');
  private alphabetReady = false;

  // Simple sign recognition based on hand positions and movements
  private signDictionary = new Map<string, string>([
    // Basic hand shapes and positions
    ['hand_open', 'hello'],
    ['hand_closed', 'yes'],
    ['hand_pointing', 'you'],
    ['hand_thumbs_up', 'good'],
    ['hand_wave', 'bye'],
    ['hand_peace', 'two'],
    ['hand_ok', 'okay'],
    ['hand_thumbs_down', 'no'],
    ['hand_fist', 'stop'],
    ['hand_open_both', 'welcome'],
    ['hand_closed_both', 'thank you'],
    // Finger count patterns
    ['hand_1_fingers', 'one finger up'],
    ['hand_2_fingers', 'two fingers up'],
    ['hand_3_fingers', 'three fingers up'],
    ['hand_4_fingers', 'four fingers up'],
    // Add more signs as needed
  ]);

  constructor() {
    this.initializeSignDetection();
    
    // Initialize with empty text
    this.currentText.next('');

    // Try loading alphabet model in background; continues gracefully if missing
    this.loadAlphabetModel().then(() => {
      console.log('ASL alphabet model loaded:', this.alphabetReady);
    });
  }

  private initializeSignDetection(): void {
    console.log('SignToTextTranslationService: Initializing sign detection');
    
    // Combine pose data and detector state
    const pose$ = this.store.select<EstimatedPose>(state => state.pose.pose);
    const detector$ = this.store.select<DetectorStateModel>(state => state.detector);

    // Debug pose data
    pose$.pipe(
      tap(pose => {
        if (pose) {
          console.log('Pose data received:', pose);
        }
      })
    ).subscribe();

    // Debug detector data
    detector$.pipe(
      tap(detector => {
        if (detector) {
          console.log('Detector data received:', detector);
        }
      })
    ).subscribe();

    combineLatest([pose$, detector$])
      .pipe(
        tap(([pose, detector]) => {
          console.log('Combined data - pose:', !!pose, 'detector:', !!detector);
        }),
        filter(([pose, detector]) => pose != null && detector != null),
        debounceTime(100), // Debounce to avoid too frequent updates
        distinctUntilChanged(([pose1, detector1], [pose2, detector2]) => 
          pose1 === pose2 && detector1.signingProbability === detector2.signingProbability
        )
      )
      .subscribe(([pose, detector]) => {
        console.log('Processing pose data:', pose, 'detector:', detector);
        this.processPoseData(pose, detector);
      });
  }

  private processPoseData(pose: EstimatedPose, detector: DetectorStateModel): void {
    if (!pose || !detector) return;

    const isCurrentlySigning = detector.isSigning;
    const confidence = detector.signingProbability;

    console.log('Processing pose data - isSigning:', isCurrentlySigning, 'confidence:', confidence);

    this.isSigning.next(isCurrentlySigning);
    this.currentConfidence.next(confidence);

    // Try to determine a concrete token (digit/known sign)
    const token = confidence > 0.08 ? this.analyzePose(pose) : null;

    const now = Date.now();
    if (token) {
      // Stability gating: accept only when the same token is seen for N frames
      if (this.lastCandidate === token) {
        this.stableCount++;
      } else {
        this.lastCandidate = token;
        this.stableCount = 1;
      }
      this.lastSeenTs = now;

      // Emit current token immediately for UI while building stability
      this.currentToken.next(token);

      if (this.stableCount >= this.requiredStableFrames) {
        this.acceptToken(token);
        // reset candidate so we don't re-accept every frame
        this.stableCount = 0;
        this.lastCandidate = null;
      }
    } else {
      this.currentToken.next('');
      // No token currently; if paused long enough, add a space once
      if (this.lastSeenTs > 0 && now - this.lastSeenTs > this.pauseMsForSpace) {
        this.acceptToken(' ');
        this.lastSeenTs = 0; // add space only once per pause
        this.stableCount = 0;
        this.lastCandidate = null;
      }
    }
  }

  private analyzePose(pose: EstimatedPose): string | null {
    console.log('Analyzing pose - leftHand:', !!pose.leftHandLandmarks, 'rightHand:', !!pose.rightHandLandmarks);
    
    // Check if we have at least one hand
    if (!pose.leftHandLandmarks && !pose.rightHandLandmarks) {
      console.log('No hand landmarks found');
      return null;
    }

    // First: try alphabet classifier if available
    const alphabet = this.classifyAlphabet(pose);
    if (alphabet) return alphabet;

    // Next: try numeric detection by counting raised fingers across both hands (0-9)
    const digit = this.detectDigitFromHands(pose);
    if (digit !== null) {
      return digit.toString();
    }

    // Try library-based gesture recognition if available
    const libGesture = this.detectWithFingerpose(pose);
    if (libGesture) {
      return libGesture;
    }

    // Analyze hand positions and shapes
    const leftHandShape = pose.leftHandLandmarks ? this.analyzeHandShape(pose.leftHandLandmarks) : 'unknown';
    const rightHandShape = pose.rightHandLandmarks ? this.analyzeHandShape(pose.rightHandLandmarks) : 'unknown';
    
    console.log('Hand shapes - left:', leftHandShape, 'right:', rightHandShape);
    
    // Analyze hand movements
    const movement = this.analyzeHandMovement(pose);

    // Combine analysis to determine the sign
    const signKey = this.determineSign(leftHandShape, rightHandShape, movement);
    console.log('Determined sign key:', signKey);
    
    const result = this.signDictionary.get(signKey) || null;
    console.log('Final sign result:', result);
    
    return result;
  }

  private detectWithFingerpose(pose: EstimatedPose): string | null {
    if (!pose.leftHandLandmarks && !pose.rightHandLandmarks) return null;
    if (!this.fingerpose) {
      // Lazy import in browser only
      try {
        // @ts-ignore
        const mod = (window as any).fingerposeLoaded ? (window as any).fingerposeLoaded : null;
        if (mod) {
          this.fingerpose = mod;
        } else {
          // dynamic import
          // @ts-ignore
          return null; // Will be available after module loads
        }
        const {GestureEstimator, Gestures} = this.fingerpose;
        const gestures: any[] = [];
        if (Gestures?.VictoryGesture) gestures.push(Gestures.VictoryGesture);
        if (Gestures?.ThumbsUpGesture) gestures.push(Gestures.ThumbsUpGesture);
        if (Gestures?.OkGesture) gestures.push(Gestures.OkGesture);
        this.gestureEstimator = new GestureEstimator(gestures);
      } catch {
        this.fingerpose = null;
      }
    }

    if (!this.gestureEstimator) return null;

    // Prefer right hand if available, else left
    const hand = pose.rightHandLandmarks || pose.leftHandLandmarks;
    if (!hand) return null;
    const landmarks = hand.map(p => ({x: p.x, y: p.y, z: p.z ?? 0}));
    try {
      const estimation = this.gestureEstimator.estimate(landmarks, 7.5);
      const best = estimation?.gestures?.sort((a: any, b: any) => b.score - a.score)[0];
      if (!best) return null;
      const name = (best.name || '').toLowerCase();
      if (name.includes('thumb') || name.includes('thumbs')) return 'good';
      if (name.includes('victory') || name.includes('v')) return 'two';
      if (name.includes('ok')) return 'okay';
      return null;
    } catch {
      return null;
    }
  }

  // Alphabet classifier using TFJS model if available under assets/models/asl-alphabet/model.json
  private classifyAlphabet(pose: EstimatedPose): string | null {
    try {
      if (!this.alphabetReady) return null;
      const hand = pose.rightHandLandmarks || pose.leftHandLandmarks;
      if (!hand || hand.length < 21) return null;

      const wrist = hand[0];
      const xs: number[] = [];
      for (let i = 0; i < 21; i++) {
        const p = hand[i];
        xs.push(p.x - wrist.x);
        xs.push(p.y - wrist.y);
      }

      const t = this.tfService.tensor2d([xs], [1, xs.length]);
      const pred = (this.alphabetModel as any).predict(t);
      const data = pred.dataSync() as Float32Array;
      const argMax = Array.from(data).reduce((best, v, idx, arr) => v > arr[best] ? idx : best, 0);
      const prob = data[argMax];
      t.dispose?.();
      pred.dispose?.();
      if (prob >= 0.6 && this.alphabetLabels[argMax]) {
        return this.alphabetLabels[argMax];
      }
      return null;
    } catch {
      return null;
    }
  }

  // Public: load alphabet model lazily
  async loadAlphabetModel(): Promise<boolean> {
    try {
      await this.tfService.load();
      const url = 'assets/models/asl-alphabet/model.json';
      this.alphabetModel = await this.tfService.loadLayersModel(url);
      this.alphabetReady = true;
      return true;
    } catch (e) {
      console.warn('ASL alphabet model not available; skipping. Place TFJS model at assets/models/asl-alphabet/model.json');
      this.alphabetReady = false;
      return false;
    }
  }

  private detectDigitFromHands(pose: EstimatedPose): number | null {
    const left = pose.leftHandLandmarks ? this.countFingersUp(pose.leftHandLandmarks) : 0;
    const right = pose.rightHandLandmarks ? this.countFingersUp(pose.rightHandLandmarks) : 0;
    const total = left + right;

    // Robustness: if no hands, return null
    if (!pose.leftHandLandmarks && !pose.rightHandLandmarks) return null;

    // 0-9 mapping by total raised fingers
    if (total >= 0 && total <= 9) {
      return total;
    }
    return null;
  }

  private analyzeHandShape(handLandmarks: any[]): string {
    if (!handLandmarks || handLandmarks.length < 21) return 'unknown';

    // Analyze finger positions to determine hand shape
    const fingersUp = this.countFingersUp(handLandmarks);
    
    console.log('Hand analysis - fingers up:', fingersUp);
    
    // Check for specific gestures first
    if (this.isOKSign(handLandmarks)) return 'hand_ok';
    if (this.isThumbDown(handLandmarks) && fingersUp === 4) return 'hand_thumbs_down';
    if (fingersUp === 1 && this.isFingerUp(handLandmarks, 4)) return 'hand_thumbs_up';
    if (fingersUp === 1 && this.isFingerUp(handLandmarks, 8)) return 'hand_pointing';
    if (fingersUp === 2 && this.isFingerUp(handLandmarks, 8) && this.isFingerUp(handLandmarks, 12)) return 'hand_peace';
    if (fingersUp === 5) return 'hand_open';
    if (fingersUp === 0) return 'hand_closed';
    
    // If we have some fingers up but not a specific pattern, describe it
    if (fingersUp > 0 && fingersUp < 5) {
      return `hand_${fingersUp}_fingers`;
    }
    
    return 'unknown';
  }

  private countFingersUp(handLandmarks: any[]): number {
    let count = 0;
    const fingerTips = [4, 8, 12, 16, 20]; // Thumb, Index, Middle, Ring, Pinky tips
    const fingerPips = [3, 6, 10, 14, 18]; // Finger PIP joints

    for (let i = 0; i < fingerTips.length; i++) {
      if (this.isFingerUp(handLandmarks, fingerTips[i], fingerPips[i])) {
        count++;
      }
    }
    return count;
  }

  private isFingerUp(handLandmarks: any[], tipIndex: number, pipIndex?: number): boolean {
    const tip = handLandmarks[tipIndex];
    const pip = pipIndex ? handLandmarks[pipIndex] : handLandmarks[tipIndex - 1];
    
    if (!tip || !pip) return false;
    
    return tip.y < pip.y; // Finger is up if tip is above PIP joint
  }

  private isThumbDown(handLandmarks: any[]): boolean {
    const thumbTip = handLandmarks[4];
    const thumbIp = handLandmarks[3];
    
    if (!thumbTip || !thumbIp) return false;
    
    return thumbTip.y > thumbIp.y; // Thumb is down if tip is below IP joint
  }

  private isOKSign(handLandmarks: any[]): boolean {
    const thumbTip = handLandmarks[4];
    const indexTip = handLandmarks[8];
    
    if (!thumbTip || !indexTip) return false;
    
    // Check if thumb and index finger are close together (touching)
    const distance = Math.sqrt(
      Math.pow(thumbTip.x - indexTip.x, 2) + 
      Math.pow(thumbTip.y - indexTip.y, 2)
    );
    
    return distance < 0.05; // Close enough to be considered touching
  }

  private isFist(handLandmarks: any[]): boolean {
    // A fist is when all fingers are curled down
    const fingersUp = this.countFingersUp(handLandmarks);
    return fingersUp === 0;
  }

  private acceptToken(token: string): void {
    // Deduplicate consecutive tokens, but allow spaces to collapse duplicates too
    const last = this.detectedSigns[this.detectedSigns.length - 1] || null;
    if (last !== token) {
      // Limit text length to reasonable window
      if (this.detectedSigns.length > 200) {
        this.detectedSigns = this.detectedSigns.slice(-200);
      }
      // Phrase shortcuts mapping for common gestures
      const mapped = this.mapTokenToPhrase(token);
      this.detectedSigns.push(mapped);
      this.lastAccepted = token;
      const fullText = this.detectedSigns.join('').replace(/\s{2,}/g, ' ');
      console.log('Accepted token:', token, 'Full text:', fullText);
      this.currentText.next(fullText);
    }
  }

  private mapTokenToPhrase(token: string): string {
    switch ((token || '').toLowerCase()) {
      case 'hand_open':
      case 'hand_open_both':
        return ' Hello ';
      case 'hand_wave':
        return ' Hello ';
      case 'hand_closed_both':
        return ' Welcome ';
      case 'good':
      case 'hand_thumbs_up':
        return ' Yes ';
      case 'hand_thumbs_down':
        return ' No ';
      case 'hand_pointing':
        return ' You ';
      case 'hand_ok':
      case 'okay':
        return ' Ok ';
      case 'hand_fist':
      case 'stop':
        return ' Stop ';
      default:
        return token;
    }
  }

  private analyzeHandMovement(pose: EstimatedPose): string {
    // This is a simplified movement analysis
    // In a real implementation, you'd track hand positions over time
    return 'static'; // For now, just return static
  }

  private determineSign(leftShape: string, rightShape: string, movement: string): string {
    // Combine hand shapes and movement to determine the sign
    if (leftShape === 'hand_open' && rightShape === 'hand_open') return 'hand_open_both';
    if (leftShape === 'hand_closed' && rightShape === 'hand_closed') return 'hand_closed_both';
    if (leftShape === 'hand_thumbs_up' || rightShape === 'hand_thumbs_up') return 'hand_thumbs_up';
    if (leftShape === 'hand_thumbs_down' || rightShape === 'hand_thumbs_down') return 'hand_thumbs_down';
    if (leftShape === 'hand_pointing' || rightShape === 'hand_pointing') return 'hand_pointing';
    if (leftShape === 'hand_peace' || rightShape === 'hand_peace') return 'hand_peace';
    if (leftShape === 'hand_ok' || rightShape === 'hand_ok') return 'hand_ok';
    if (leftShape === 'hand_fist' || rightShape === 'hand_fist') return 'hand_fist';
    
    // If both hands are doing the same thing, use that
    if (leftShape === rightShape && leftShape !== 'unknown') return leftShape;
    
    // Otherwise, prioritize the more specific hand shape
    return leftShape !== 'unknown' ? leftShape : rightShape;
  }

  // Public methods for components to use
  getCurrentText(): Observable<string> {
    return this.currentText.asObservable();
  }

  getCurrentToken(): Observable<string> {
    return this.currentToken.asObservable();
  }

  getCurrentConfidence(): Observable<number> {
    return this.currentConfidence.asObservable();
  }

  getIsSigning(): Observable<boolean> {
    return this.isSigning.asObservable();
  }

  getSignToTextResult(): Observable<SignToTextResult> {
    return combineLatest([
      this.currentText,
      this.currentConfidence,
      this.isSigning
    ]).pipe(
      map(([text, confidence, signing]) => ({
        text,
        confidence,
        timestamp: Date.now()
      }))
    );
  }

  // Method to add custom signs to the dictionary
  addCustomSign(signKey: string, text: string): void {
    this.signDictionary.set(signKey, text);
  }

  // Method to clear current text
  clearText(): void {
    this.currentText.next('');
    this.detectedSigns = [];
  }

  // Method to get all detected signs
  getDetectedSigns(): string[] {
    return [...this.detectedSigns];
  }
}
