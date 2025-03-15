import type { Config } from "tailwindcss";

const config: Config = {
  // Ensures that Tailwind processes all relevant source files
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require("@tailwindcss/typography"),
    require("@tailwindcss/forms")
  ],
};

export default config;
