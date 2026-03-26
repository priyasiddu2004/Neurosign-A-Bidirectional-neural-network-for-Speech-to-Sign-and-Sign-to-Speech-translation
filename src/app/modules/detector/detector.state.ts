import {inject, Injectable} from '@angular/core';
import {Action, NgxsOnInit, State, StateContext, Store} from '@ngxs/store';
import {DetectorService} from './detector.service';
import {DetectSigning} from './detector.actions';
import {filter, first, tap} from 'rxjs/operators';
import {EstimatedPose} from '../pose/pose.state';
import {Observable} from 'rxjs';

export interface DetectorStateModel {
  signingProbability: number;
  isSigning: boolean;
}

const initialState: DetectorStateModel = {
  signingProbability: 0,
  isSigning: false,
};

@Injectable()
@State<DetectorStateModel>({
  name: 'detector',
  defaults: initialState,
})
export class DetectorState implements NgxsOnInit {
  private store = inject(Store);
  private detector = inject(DetectorService);

  detectSign = false;
  pose$: Observable<EstimatedPose>;
  detectSign$: Observable<boolean>;

  constructor() {
    this.pose$ = this.store.select<EstimatedPose>(state => state.pose.pose);
    this.detectSign$ = this.store.select<boolean>(state => state.settings.detectSign);
  }

  ngxsOnInit({dispatch}: StateContext<any>): void {
    console.log('DetectorState: Initializing');
    
    // Load model once setting turns on
    this.detectSign$
      .pipe(
        tap(detectSign => console.log('DetectSign setting changed:', detectSign)),
        filter(Boolean),
        first(),
        tap(() => {
          console.log('Loading detector model...');
          this.detector.loadModel();
        })
      )
      .subscribe();
    this.detectSign$.subscribe(detectSign => {
      console.log('DetectSign updated:', detectSign);
      this.detectSign = detectSign;
    });

    this.pose$
      .pipe(
        tap(pose => console.log('Pose received in detector:', !!pose)),
        filter(Boolean),
        tap(() => console.log('DetectSign status:', this.detectSign)),
        filter(() => this.detectSign), // Only run if needed
        tap((pose: EstimatedPose) => {
          console.log('Dispatching DetectSigning action');
          dispatch(new DetectSigning(pose));
        })
      )
      .subscribe();
  }

  @Action(DetectSigning)
  detectSigning({getState, patchState}: StateContext<DetectorStateModel>, {pose}: DetectSigning): void {
    console.log('DetectSigning action triggered');
    const signingProbability = this.detector.detect(pose);
    console.log('Signing probability:', signingProbability);
    patchState({
      signingProbability,
      isSigning: signingProbability > 0.5,
    });
    console.log('Detector state updated - isSigning:', signingProbability > 0.5);
  }
}
