// Thin wrapper over the backend routes the terminal uses.

async function get(path) {
  let response;
  try {
    response = await fetch(path, { headers: { Accept: 'application/json' } });
  } catch (error) {
    throw new Error('Cannot reach the Niveshak API. Check that the server is running.');
  }
  if (!response.ok) {
    let detail = '';
    try { detail = (await response.json()).detail || ''; } catch (error) { /* not JSON */ }
    throw new Error(detail || `The server answered ${response.status} for ${path}`);
  }
  return response.json();
}

const withAt = (path, at) => (at ? `${path}?at=${encodeURIComponent(at)}` : path);

export const api = {
  snapshot: (at) => get(withAt('/api/v1/niveshak/snapshot', at)),
  stock: (symbol, at) => get(withAt(`/api/v1/niveshak/stocks/${encodeURIComponent(symbol)}`, at)),
  lessons: () => get('/api/v1/niveshak/lessons'),
  health: () => get('/health'),
  risk: () => get('/risk/status'),
};
