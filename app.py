import gradio as gr
import tensorflow as tf

model = tf.keras.models.load_model('convolution_model.h5')


def reconnaissance_chiffre(image):
    if image is not None:

        image = image.reshape((1, 28, 28, 1)).astype('float32') / 255

        prediction = model.predict(image)

        resultats = {str(i): float(prediction[0][i]) for i in range(10)}

        return resultats
    else:
        return ''


iface = gr.Interface(
    fn=reconnaissance_chiffre,
    inputs=gr.Image(
        shape=(28, 28),
        image_mode='L',
        invert_colors=True,
        source='canvas'),
    outputs=gr.Label(num_top_classes=3, color="green"),
    live=True,
    title="handwritten digit recognition application",
)

iface.launch()
