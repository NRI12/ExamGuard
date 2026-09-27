interface StorageLike {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export interface TokenStore {
  get(): string | null
  set(token: string): void
  clear(): void
}

const tokenKey = 'examguard_access_token'

export function createTokenStore(storage: StorageLike): TokenStore {
  return {
    get: () => storage.getItem(tokenKey),
    set: (token) => storage.setItem(tokenKey, token),
    clear: () => storage.removeItem(tokenKey),
  }
}

export const browserTokenStore: TokenStore = {
  get: () => window.localStorage.getItem(tokenKey),
  set: (token) => window.localStorage.setItem(tokenKey, token),
  clear: () => window.localStorage.removeItem(tokenKey),
}
