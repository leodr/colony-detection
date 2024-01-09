import { Canvas } from "./Canvas";
import { ImageDropArea } from "./ImageDropArea";

function App() {
  return (
    <div className="p-4 bg-gray-950 flex min-h-screen text-white">
      <nav className="w-96">
        <ImageDropArea></ImageDropArea>
      </nav>
      <Canvas></Canvas>
    </div>
  );
}

export default App;
