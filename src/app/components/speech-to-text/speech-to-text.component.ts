import {Component, Input, OnChanges, OnInit, output, SimpleChanges, OnDestroy, inject} from '@angular/core';
import {Subject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {BaseComponent} from '../base/base.component';
import {MatTooltipModule, TooltipPosition} from '@angular/material/tooltip';
import {IonButton, IonIcon} from '@ionic/angular/standalone';
import {TranslocoDirective} from '@jsverse/transloco';
import {addIcons} from 'ionicons';
import {micOutline, stopCircleOutline, micOffOutline} from 'ionicons/icons';
import {SpeechService} from '../../core/services/speech.service';

@Component({
  selector: 'app-speech-to-text',
  templateUrl: './speech-to-text.component.html',
  styleUrls: ['./speech-to-text.component.css'],
  imports: [IonButton, IonIcon, MatTooltipModule, TranslocoDirective],
})
export class SpeechToTextComponent extends BaseComponent implements OnInit, OnChanges, OnDestroy {
  @Input() lang = 'en';
  readonly changeText = output<string>();
  @Input() matTooltipPosition: TooltipPosition = 'above';

  private speechService = inject(SpeechService);
  private destroy$ = new Subject<void>();

  supportError: string | null = null;
  isRecording = false;

  constructor() {
    super();

    addIcons({stopCircleOutline, micOutline, micOffOutline});
  }

  ngOnInit(): void {
    console.log('SpeechToTextComponent: Initializing...');
    
    // Set language
    this.speechService.setLanguage(this.lang);
    
    // Subscribe to speech service observables
    this.speechService.getCurrentText()
      .pipe(takeUntil(this.destroy$))
      .subscribe(text => {
        if (text.trim()) {
          console.log('Speech text received:', text);
          this.changeText.emit(text);
        }
      });

    this.speechService.getIsRecording()
      .pipe(takeUntil(this.destroy$))
      .subscribe(recording => {
        this.isRecording = recording;
      });

    this.speechService.getError()
      .pipe(takeUntil(this.destroy$))
      .subscribe(error => {
        this.supportError = error;
      });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes.lang) {
      this.speechService.setLanguage(this.lang);
    }
  }

  override ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.speechService.destroy();
  }

  async start(): Promise<void> {
    console.log('Starting speech recognition...');
    const success = await this.speechService.startRecording();
    if (!success) {
      console.error('Failed to start speech recognition');
    }
  }

  stop(): void {
    console.log('Stopping speech recognition...');
    this.speechService.stopRecording();
  }
}