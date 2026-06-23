import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  build: {
    lib: {
      entry: path.resolve(__dirname, "plugin-shell/src/index.tsx"),
      name: "Tumor3DPlugin",
      fileName: "plugin-shell",
      formats: ["es", "umd"],
    },
    rollupOptions: {
      external: ["react", "react-dom", "three", "@react-three/fiber", "@react-three/drei"],
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
          three: "THREE",
          "@react-three/fiber": "ReactThreeFiber",
          "@react-three/drei": "ReactThreeDrei",
        },
      },
    },
    outDir: "../dist/plugin-shell",
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      "@shared": path.resolve(__dirname, "../shared/schemas/typescript"),
      "@viewer": path.resolve(__dirname, "viewer/src"),
    },
  },
});
