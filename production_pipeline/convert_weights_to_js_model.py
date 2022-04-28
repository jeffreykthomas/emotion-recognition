from training_models.cyclegan import get_model as get_cyclegan_model
from training_models.emotion_recognition_training import get_model as get_recognizer_model
from training_models.conditional_gan import get_model as get_cgan_model
from subprocess import call
import os

weight_file_root = '/Volumes/Pegasus_R4i/cycle_gan/128_weights/'
best_weights = {
    'recognizer': 'results/recognizer/128_reduced_True_augmented_False_7cat/model.h5',
    'conditional_gan': 'results/conditional_gan/checkpoint',
    'happy-anger': weight_file_root + 'anger_happiness/checkpoints/cyclegan_checkpoints_171',
    'happy-contempt': weight_file_root + 'contempt_happiness/checkpoints/cyclegan_checkpoints_002',
    'happy-disgust': weight_file_root + 'disgust_happiness/checkpoints/cyclegan_checkpoints_007',
    'happy-fear': weight_file_root + 'fear_happiness/checkpoints/cyclegan_checkpoints_003',
    'happy-neutral': weight_file_root + 'neutral_happiness/checkpoints/cyclegan_checkpoints_003',
    'happy-sadness': weight_file_root + 'sadness_happiness/checkpoints/cyclegan_checkpoints_233',
    'happy-surprise': weight_file_root + 'surprise_happiness/checkpoints/cyclegan_checkpoints_001'
}

cyclegan_model, _, _ = get_cyclegan_model((128, 128, 3))
recognizer_model = get_recognizer_model(128, 7)
conditional_gan_model = get_cgan_model(64, 8)

for (model, weight) in best_weights.items():
    output_folder = 'web-app/public/models/'
    save_dir = 'production_pipeline/models/'

    def save_model(model_type, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        if model_type == cyclegan_model:
            model_type.load_weights(weight).expect_partial()
            model_type.gen_F.save(save_directory)
        elif model_type == conditional_gan_model:
            model_type.load_weights(weight).expect_partial()
            model_type.generator.save(save_directory)
        else:
            model_type.load_weights(weight)
            model_type.save(save_directory)

    if model.split('-')[0] == 'happy':
        save_dir = save_dir + 'cyclegan/' + model
        save_model(cyclegan_model, save_dir)
        output_folder = output_folder + 'cyclegan/' + model

    elif model == 'recognizer':
        save_dir = save_dir + model
        save_model(recognizer_model, save_dir)
        output_folder = output_folder + model

    else:
        save_dir = save_dir + model
        save_model(conditional_gan_model, save_dir)
        output_folder = output_folder + model

    var = [
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        save_dir,
        output_folder
    ]
    call(var)
