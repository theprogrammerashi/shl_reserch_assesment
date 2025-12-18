/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'shl-blue': '#00bceb', 
        'shl-dark': '#2d3748',
      }
    },
  },
  plugins: [],
}
