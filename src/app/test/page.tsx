export default function TestPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md mx-auto bg-white rounded-xl shadow-macos overflow-hidden md:max-w-2xl">
        <div className="p-8">
          <div className="tracking-wide text-sm text-indigo-500 font-semibold">
            Tailwind Test
          </div>
          <h1 className="mt-4 text-3xl font-bold tracking-tight text-gray-900">
            Testing Tailwind Classes
          </h1>
          <p className="mt-2 text-gray-500">
            If you can see this card with proper styling, Tailwind is working correctly.
          </p>
          <div className="mt-4 space-y-4">
            <button className="macos-button-primary">
              Primary Button
            </button>
            <button className="macos-button">
              Secondary Button
            </button>
            <div className="grid grid-cols-2 gap-4">
              <div className="macos-panel p-4">Panel Item 1</div>
              <div className="macos-panel p-4">Panel Item 2</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
