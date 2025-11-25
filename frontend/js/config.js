// static/js/config.js
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
  navigator.userAgent
);

const config = {
  // Hiển thị gì trên overlay
  showPersons: true,
  showFaces: true,
  showConfidence: true,
  showFaceNames: true,

  // Màu khung vẽ
//   personColor: "#e74c3c",
  faceColor: "#2ecc71",

  // API endpoints
  serverUrl: "/process_frame",
  enrollUrl: "/enroll_face",
  unlockUrl: "/unlock",

  // FPS
  frameRate: isMobile ? 15 : 25,

  // Device info
  isMobile,

  // Label
  desktopLabelFontSize: 10,
  mobileLabelFontSize: 18,
  labelPadding: 4,
  labelMargin: 6,
  borderWidth: 2,
};

export default config;
