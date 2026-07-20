import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AboutPage } from "./pages/AboutPage";
import { ComparisonPage } from "./pages/ComparisonPage";
import { HomePage } from "./pages/HomePage";
import { PredictionPage } from "./pages/PredictionPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="prediction" element={<PredictionPage />} />
        <Route path="comparison" element={<ComparisonPage />} />
        <Route path="about" element={<AboutPage />} />
      </Route>
    </Routes>
  );
}
