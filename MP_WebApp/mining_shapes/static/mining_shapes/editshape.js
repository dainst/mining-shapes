let canvas, ctx, clearButton, saveButton, image;
let width, height;
const pos = { x: 0, y: 0 };
let polygon = [];

document.addEventListener('DOMContentLoaded', function () {

    clearButton = document.getElementById('clear');
    clearButton.addEventListener('click', clear);
    saveButton = document.getElementById('save');
    saveButton.addEventListener('click', editProfile);
    image = document.getElementById('image')
    width = image.width;
    height = image.height;
    initCanvas();
});

const initCanvas = () => {

    canvas = document.getElementById('canvas');
    canvas.addEventListener('click', setPolygonPoint);
    canvas.addEventListener('oncontextmenu', removePolygonPoint);
    ctx = canvas.getContext('2d');

    canvas.width = width;
    canvas.height = height;
    erase();

};

const clear = () => {
    erase();
    polygon = [];
}

const erase = () => {

    ctx.clearRect(0, 0, width, height);
}

const getPosition = (e) => {

    const x = e.clientX - canvas.getBoundingClientRect().left;
    const y = e.clientY - canvas.getBoundingClientRect().top;
    return { x, y }
}

const draw = (e) => {

    if (polygon.length <= 1) return;

    erase();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgb(227, 252, 3)";

    for (i = 0; i < polygon.length - 1; i++) {
        drawLine(polygon[i], polygon[i + 1]);
    }
    drawLine(polygon[polygon.length - 1], polygon[0]);

}

const drawLine = (pos1, pos2) => {

    ctx.beginPath();
    ctx.setLineDash([3, 3])
    ctx.moveTo(pos1.x, pos1.y);
    ctx.lineTo(pos2.x, pos2.y);
    ctx.stroke();
}

const setPolygonPoint = (e) => {

    const position = getPosition(e);
    polygon.push(position);
    draw();

}

const removePolygonPoint = (e) => {

    e.preventDefault();
    polygon.pop();
}

const editProfile = () => {

    fetch(`/editshape/${saveButton.dataset.shapeid}`, {
        method: 'PUT',
        body: JSON.stringify({ polygon })
    })
        .then((response) => response.json())
        .then((data) => {
            showEditResults(data)
        });
}

const showEditResults = (data) => {
    clear();

    const message = document.getElementById('message');
    message.innerHTML = `<h2 class="alert alert-success alert-dismissible fade show">
                            ${data.message}
                        </h2>`;

    const imageRow = document.getElementById('image_row');
    imageRow.innerHTML = `<div class="col">
                                <img src=${data.segmentedUrl} />
                            </div>
                            <div class="col">
                                <img src=${data.imageUrl} />
                            </div>`;

    const actionsCol = document.getElementById('actions_col');
    actionsCol.innerHTML = `<a href=/sessionresult/${data.sessionId} class="btn btn-success mt-1">Return to session</a>`;

}