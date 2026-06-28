// Safe localStorage access. The providers read their config from localStorage,
// but this code is also exercised outside a browser (unit tests, SSR-style
// hosts) where `localStorage` may be absent or a non-functional stub. These
// helpers degrade to no-ops instead of throwing.

function store(): Storage | null {
  try {
    if (typeof localStorage !== "undefined" && typeof localStorage.getItem === "function") {
      return localStorage;
    }
  } catch {
    // Accessing localStorage can throw (e.g. disabled cookies / sandboxed).
  }
  return null;
}

export function getLocal(key: string): string | null {
  try {
    return store()?.getItem(key) ?? null;
  } catch {
    return null;
  }
}
