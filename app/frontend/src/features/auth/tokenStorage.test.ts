import { describe, expect, it } from 'vitest'

import { createTokenStore } from './tokenStorage'

function memoryStorage() {
  const values = new Map<string, string>()
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
  }
}

describe('token storage', () => {
  it('stores and clears the access token', () => {
    const store = createTokenStore(memoryStorage())

    expect(store.get()).toBeNull()
    store.set('signed-token')
    expect(store.get()).toBe('signed-token')
    store.clear()
    expect(store.get()).toBeNull()
  })
})
