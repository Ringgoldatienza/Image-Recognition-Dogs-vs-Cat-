library(tensorflow)
library(keras)
library(reticulate)

#Install cross-platform file system operations
install.packages("fs")
library(fs)

#Download dogs vs cats dataset on kaggle
dir_create("~/.kaggle")
file_move("~/Downloads/kaggle.json", "~/.kaggle/")
file_chmod("~/.kaggle/kaggle.json", "0600")
reticulate::py_install("kaggle", pip = TRUE)
system('kaggle competitions download -c dogs-vs-cats')

#Install Zip app
install.packages("zip")
library(zip)

#Unzip dogs vs cats
zip::unzip('dogs-vs-cats.zip', exdir = "dogs-vs-cats", files = "train.zip")
zip::unzip("dogs-vs-cats/train.zip", exdir = "dogs-vs-cats")

#Copy images to training, validation and test directories
original_dir <- path("dogs-vs-cats/train")
new_base_dir <- path("cats_vs_dogs_small")

make_subset <- function(subset_name,
                        start_index, end_index) {
  for (category in c("dog", "cat")) {
    file_name <- glue::glue("{category}.{ start_index:end_index }.jpg")
    dir_create(new_base_dir / subset_name / category)
    file_copy(original_dir / file_name,
              new_base_dir / subset_name / category / file_name)
  }
}

#Subset train, validation and test sets
make_subset("train", start_index = 1, end_index = 1000)
make_subset("validation", start_index = 1001, end_index = 1500)
make_subset("test", start_index = 1501, end_index = 2500)

#Initiate a small convet for dogs vs cats classification
inputs <- layer_input(shape = c(180, 180, 3))
outputs <- inputs %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)


#Configuring the model for training
model %>% compile(loss = "binary_crossentropy",
                  optimizer = "rmsprop",
                  metrics = "accuracy")

#Use image_data_set_from_directory to read images
train_dataset <-
  image_dataset_from_directory(new_base_dir / "train",
                               image_size = c(180, 180),
                               batch_size = 32)
validation_dataset <-
  image_dataset_from_directory(new_base_dir / "validation",
                               image_size = c(180, 180),
                               batch_size = 32)
test_dataset <-
  image_dataset_from_directory(new_base_dir / "test",
                               image_size = c(180, 180),
                               batch_size = 32) 

#Displaying the shapes of the data and labels yielded by the Dataset
c(data_batch, labels_batch) %<-% iter_next(as_iterator(train_dataset))
data_batch$shape
labels_batch$shape

#Fit model using a TensorFlow Dataset
callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_from_scratch.keras",
    save_best_only = TRUE,
    monitor = "val_loss"
  )
)

history <- model %>%
  fit(
    train_dataset,
    epochs = 30,
    validation_data = validation_dataset,
    callbacks = callbacks
  )

#Display curves of loss and accuracy during training
plot(history)

#Evaluate the model on the test set
test_model <- load_model_tf("convnet_from_scratch.keras")

result <- evaluate(test_model, test_dataset)

cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))

