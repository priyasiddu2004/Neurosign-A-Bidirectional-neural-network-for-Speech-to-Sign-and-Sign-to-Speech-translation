import {HttpClient} from '@angular/common/http';
import {Translation, TRANSLOCO_SCOPE, TranslocoLoader} from '@jsverse/transloco';
import {inject, Injectable, PLATFORM_ID} from '@angular/core';
import {isPlatformBrowser} from '@angular/common';

import {catchError, Observable, of} from 'rxjs';

@Injectable({providedIn: 'root'})
export class HttpLoader implements TranslocoLoader {
  private http = inject(HttpClient);
  private platformId = inject(PLATFORM_ID);
  private isBrowser = isPlatformBrowser(this.platformId);

  getTranslation(langPath: string): Observable<Translation> {
    if (!this.isBrowser) {
      // Avoid fetching during SSR; return empty translations to prevent crashes
      return of({} as Translation);
    }

    // Use a relative path so the dev server serves from /assets correctly
    const assetPath = `assets/i18n/${langPath}.json`;
    return this.http.get<Translation>(assetPath).pipe(
      catchError(err => {
        console.error(`Couldn't load translation file '${assetPath}'`, err);
        // Return empty object so app continues even if optional scopes are missing
        return of({} as Translation);
      })
    );
  }
}

export const translocoScopes = {
  provide: TRANSLOCO_SCOPE,
  // Only load the root app translations. Optional scope files are removed.
  useValue: [''],
};
