import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: string | null;
}

export class ViewerErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error: error.message || "3D viewer failed to load" };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[MeshViewer]", error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <p className="mesh-viewer__error">
          Could not load 3D model: {this.state.error}. Check that RunPod is running and assets are
          reachable.
        </p>
      );
    }
    return this.props.children;
  }
}
