let canvas, ctx, clearButton;
let width, height;
const pos = { x: 0, y: 0 };

document.addEventListener('DOMContentLoaded', function () {
    clearButton = document.getElementById('clear');
    clearButton.addEventListener('click', erase);
    width = document.getElementById('width').textContent;
    height = document.getElementById('height').textContent;
    initCanvas();
});

const initCanvas = () => {
    canvas = document.getElementById('canvas');
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mousedown", setPosition);
    canvas.addEventListener("mouseenter", setPosition);
    ctx = canvas.getContext('2d');

    canvas.width = width;
    canvas.height = height;
    erase();

};

const erase = () => {
    const background = new Image();
    background.src = document.getElementById('url').textContent;
    background.onload = function () {
        ctx.drawImage(background, 0, 0);
    }
}

const setPosition = (e) => {
    pos.x = e.clientX - canvas.getBoundingClientRect().left;
    pos.y = e.clientY - canvas.getBoundingClientRect().top;
}

const draw = (e) => {
    if (e.buttons != 1) return;
    ctx.beginPath();
    ctx.lineWidth = 5;
    //ctx.fillStyle = 'black';
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'blue';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}