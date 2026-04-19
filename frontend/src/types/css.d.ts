declare module '*.module.css' {
  const classes: { readonly [key: string]: string };
  export default classes;
}

// Side-effect CSS imports (e.g. `import './App.css'`,
// `import '@xyflow/react/dist/style.css'`). Required under
// `noUncheckedSideEffectImports`.
declare module '*.css';
