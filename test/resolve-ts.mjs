// Minimal ESM resolver hook: the source uses extensionless, bundler-style
// imports ("./foo"), which Node's native resolver doesn't map to ".ts". This
// hook appends ".ts" for relative specifiers that lack an extension so the
// unit tests can import the real source modules directly (Node ≥ 22.6 strips
// the TS types itself). It only touches relative paths; bare specifiers pass
// straight through.
export async function resolve(specifier, context, next) {
  const isRelative = specifier.startsWith("./") || specifier.startsWith("../");
  const hasExt = /\.[cm]?[jt]s$|\.json$/i.test(specifier);
  if (isRelative && !hasExt) {
    try {
      return await next(specifier + ".ts", context);
    } catch {
      // Fall through to the default resolution (e.g. directory index).
    }
  }
  return next(specifier, context);
}
