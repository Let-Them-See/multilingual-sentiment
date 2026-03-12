/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Project 6-color palette
        mint: "#F1F6F4",
        gold: "#FFC801",
        teal: {
          DEFAULT: "#114C5A",
          dark: "#172B36",
        },
        sage: "#D9E8E2",
        orange: "#FF9932",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        card: "0 2px 12px rgba(23,43,54,0.08)",
        "card-hover": "0 6px 24px rgba(23,43,54,0.14)",
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.5rem",
      },
    },
  },
  plugins: [],
};
