import * as tf from '@tensorflow/tfjs-node';

export default class ResNet50 {
    constructor() {
        this.model;
        this.modelPath = './model/model.json';
    }

    initialize = async () => {
        this.model = await tf.loadLayersModel(this.modelPath);
    };

    static create = async () => {
        const model = new ResNet50();
        await model.initialize();
        return model;
    };

    predict = (canvas) => {
        const raw = tf.browser.fromPixels(canvas, 3);
        const resized = tf.image.resizeBilinear(raw, [512, 512]);
        const tensor = resized.expandDims(0);
        return this.model.predict(tensor).reshape([-1]);
    };
}