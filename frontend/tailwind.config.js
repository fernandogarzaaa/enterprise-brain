/** Build: tailwindcss -c frontend/tailwind.config.js -i frontend/input.css -o enterprise_brain/static/app.css --minify */
module.exports = {
  content: ["./enterprise_brain/templates/**/*.html"],
  theme: { extend: {} },
  plugins: [],
};
