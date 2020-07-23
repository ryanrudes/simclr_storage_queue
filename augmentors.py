from tensorpack.dataflow.imgaug import *
import cv2

def get_queue_entry_augmentor(strength, p_color_jitter, p_gaussian_blur):
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength

  augmentor = AugmentorList([
                             RandomApplyAug(
                                 RandomOrderAug([
                                                 AugmentorList([
                                                                BrightnessScale((max([1.0 - brightness, 0]), 1.0 + brightness)),
                                                                Clip(min = 0, max = 1)
                                                ]),

                                                AugmentorList([
                                                                Contrast((1.0 - contrast, 1.0 + contrast)),
                                                                Clip(min = 0, max = 1)
                                                ]),

                                                AugmentorList([
                                                                Saturation(saturation),
                                                                Clip(min = 0, max = 1)
                                                ]),

                                                AugmentorList([
                                                                Hue(range = (-hue, hue)),
                                                                Clip(min = 0, max = 1)
                                                ])
                                 ]),
                                 prob = p_color_jitter
                             ),

                             RandomApplyAug(GaussianBlur(size_range = (21.4, 22.4), sigma_range = (0.1, 2)), p_gaussian_blur)
  ])

  return augmentor

def get_queue_augmentor(strength, p_color_jitter, p_gaussian_blur):
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength

  augmentor = AugmentorList([
                             RandomApplyAug(
                                 RandomOrderAug([
                                                 AugmentorList([
                                                                BrightnessScale((max([1.0 - brightness, 0]), 1.0 + brightness)),
                                                                Clip(min = 0, max = 1)
                                                ]),

                                                AugmentorList([
                                                                Contrast((1.0 - contrast, 1.0 + contrast)),
                                                                Clip(min = 0, max = 1)
                                                ]),

                                                AugmentorList([
                                                                Saturation(saturation),
                                                                Clip(min = 0, max = 1)
                                                ]),

                                                AugmentorList([
                                                                Hue(range = (-hue, hue)),
                                                                Clip(min = 0, max = 1)
                                                ])
                                 ]),
                                 prob = p_color_jitter
                             ),
                              
                             RandomApplyAug(GaussianBlur(size_range = (21.4, 22.4), sigma_range = (0.1, 2)), p_gaussian_blur)
  ])

  return augmentor

def get_training_augmentor(strength, p_color_jitter, p_gaussian_blur, p_flip, p_grayscale):
  crop_area_range = (0.08, 1.0)
  crop_height, crop_width = 224, 224
  aspect_ratio = crop_width / crop_height
  aspect_ratio_range = (3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio)

  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength

  augmentor = AugmentorList([
                             GoogleNetRandomCropAndResize(
                                 crop_area_fraction = crop_area_range,
                                 aspect_ratio_range = aspect_ratio_range,
                                 target_shape = crop_height,
                                 interp = cv2.INTER_CUBIC
                             ),
                             
                             Flip(horiz = True, prob = p_flip),
                             
                             RandomApplyAug(
                                 RandomOrderAug([
                                                 AugmentorList([
                                                                BrightnessScale((max([1.0 - brightness, 0]), 1.0 + brightness)),
                                                                Clip(min = 0, max = 1)
                                                ]),

                                                AugmentorList([
                                                                Contrast((1.0 - contrast, 1.0 + contrast)),
                                                                Clip(min = 0, max = 1)
                                                ]),

                                                AugmentorList([
                                                                Saturation(saturation),
                                                                Clip(min = 0, max = 1)
                                                ]),

                                                AugmentorList([
                                                                Hue(range = (-hue, hue)),
                                                                Clip(min = 0, max = 1)
                                                ])
                                 ]),
                                 prob = p_color_jitter
                             ),
                              
                             RandomApplyAug(Grayscale(rgb = True, keepshape = True), prob = p_grayscale),
                              
                             Clip(min = 0, max = 1),

                             RandomApplyAug(GaussianBlur(size_range = (21.4, 22.4), sigma_range = (0.1, 2)), p_gaussian_blur)
  ])

  return augmentor

def get_fine_tuning_augmentor():
  crop_area_range = (0.08, 1.0)
  crop_height, crop_width = 224, 224
  aspect_ratio = crop_width / crop_height
  aspect_ratio_range = (3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio)

  augmentor = AugmentorList([
                             GoogleNetRandomCropAndResize(
                                 crop_area_fraction = crop_area_range,
                                 aspect_ratio_range = aspect_ratio_range,
                                 target_shape = crop_height,
                                 interp = cv2.INTER_CUBIC
                             ),
                             
                             Flip(horiz = True, prob = 0.5),
  ])

  return augmentor

def get_inference_augmentor():
  center_crop_shape = (224, 224)
  
  augmentor = AugmentorList([
                             CenterCrop(center_crop_shape)
  ])

  return augmentor
