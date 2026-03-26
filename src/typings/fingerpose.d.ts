declare module 'fingerpose' {
  export class GestureEstimator {
    constructor(gestures: any[]);
    estimate(landmarks: Array<{x:number;y:number;z?:number}>, minConfidence: number): {gestures: Array<{name: string; score: number}>};
  }

  export const Gestures: {
    VictoryGesture: any;
    ThumbsUpGesture: any;
    OkGesture?: any;
  };
}


