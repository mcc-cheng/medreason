import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        accent: "#34D399",
        "accent-dim": "#059669",
        "accent-muted": "rgba(52, 211, 153, 0.08)",
        bg: {
          DEFAULT: "#09090B",
          raised: "#0F0F12",
        },
        surface: {
          DEFAULT: "#131316",
          hover: "#1A1A1F",
          border: "#1F1F24",
          "border-light": "rgba(255,255,255,0.06)",
        },
        "zero-shot": "#64748B",
        memory: "#34D399",
        text: {
          primary: "#FAFAFA",
          secondary: "#71717A",
          tertiary: "#3F3F46",
        },
      },
      fontFamily: {
        sans: ['"Geist"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', '"Geist Mono"', "monospace"],
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
        "4xl": "2rem",
      },
      transitionTimingFunction: {
        spring: "cubic-bezier(0.32, 0.72, 0, 1)",
        "spring-heavy": "cubic-bezier(0.16, 1, 0.3, 1)",
        "out-expo": "cubic-bezier(0.19, 1, 0.22, 1)",
      },
      keyframes: {
        "pulse-soft": {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
        "shimmer": {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "float": {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-4px)" },
        },
      },
      animation: {
        "pulse-soft": "pulse-soft 2.5s ease-in-out infinite",
        shimmer: "shimmer 3s ease-in-out infinite",
        float: "float 4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
