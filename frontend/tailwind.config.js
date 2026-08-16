/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        storm: {
          950: '#07090f',
          900: '#0d1117',
          800: '#161b27',
          700: '#1f2637',
          600: '#2d3748',
        },
      },
    },
  },
  plugins: [],
}
