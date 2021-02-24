
document.addEventListener('DOMContentLoaded', function () {
    intiCanvas();
});

const intiCanvas = () => {
    const canvas = document.getElementById("canvas");
    let ctx = canvas.getContext("2d");
    canvas.width = document.getElementById('width').textContent;
    canvas.height = document.getElementById('height').textContent;

    const background = new Image();
    background.src = document.getElementById('url').textContent;
    background.onload = function () {
        ctx.drawImage(background, 0, 0);
    }
};