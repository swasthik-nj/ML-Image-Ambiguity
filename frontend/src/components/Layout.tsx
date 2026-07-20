import { NavLink, Outlet } from "react-router-dom";

const links = [
  { to: "/", label: "Home", end: true },
  { to: "/prediction", label: "Prediction" },
  { to: "/comparison", label: "Comparison" },
  { to: "/about", label: "About" },
];

export function Layout() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-ink/10 bg-fog/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 md:px-6">
          <NavLink to="/" className="group flex items-baseline gap-2">
            <span className="font-display text-xl font-extrabold tracking-tight text-ink md:text-2xl">
              Ambiguity Lens
            </span>
            <span className="hidden text-xs text-ink/50 sm:inline">
              research demo
            </span>
          </NavLink>
          <nav className="flex flex-wrap items-center gap-1 sm:gap-2">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.end}
                className={({ isActive }) =>
                  [
                    "rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-ink text-fog"
                      : "text-ink/70 hover:bg-ink/5 hover:text-ink",
                  ].join(" ")
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8 md:px-6 md:py-12">
        <Outlet />
      </main>
      <footer className="border-t border-ink/10 py-6 text-center text-sm text-ink/50">
        Explainable Image Ambiguity Prediction · FastAPI + React
      </footer>
    </div>
  );
}
