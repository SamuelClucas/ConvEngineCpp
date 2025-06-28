CXX := g++
CXXFLAGS := -DGL_SILENCE_DEPRECATION -std=c++20 -Wall -O2

libdir := -L/opt/homebrew/opt/glfw/lib -L/opt/homebrew/opt/boost/lib -L/opt/homebrew/opt/opencv/lib -L/opt/homebrew/opt/onnxruntime/lib
incl := -I/opt/homebrew/opt/boost/include -I/opt/homebrew/opt/opencv/include -I/opt/homebrew/opt/opencv/include/opencv4 -I/opt/homebrew/opt/onnxruntime/include -Iexternal/imgui -Iexternal/implot -I/opt/homebrew/include -Iexternal/imgui/backends
lib := -lglfw -lboost_system -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_core -lonnxruntime
mac_frameworks := -framework OpenGL

imgui_src := \
    external/imgui/imgui.cpp \
    external/imgui/imgui_draw.cpp \
    external/imgui/imgui_tables.cpp \
    external/imgui/imgui_widgets.cpp \
    external/imgui/backends/imgui_impl_glfw.cpp \
    external/imgui/backends/imgui_impl_opengl3.cpp

implot_src := \
    external/implot/implot.cpp \
    external/implot/implot_items.cpp

# === Source files ===
src_common :=  src/train_refactor.cpp src/kernel.cpp
inc := -Iinclude/

# Colors for pretty output
green := \033[1;32m
blue := \033[0;34m
black := \033[0;30m

# === Executable: DigitalScribe (GUI) ===
#scribe_src := src/main.cpp $(src_common) $(imgui_src) $(implot_src)
#scribe_obj := $(scribe_src:.cpp=.o)
#scribe_bin := DigitalScribe

# === Executable: record_images (data label tool) ===
#record_src := src/record_images.cpp
#ecord_obj := $(record_src:.cpp=.o)
#record_bin := record_images

# === Refactor: label, sample, dataset, train ===
trainRefactor_src := $(src_common) #src/train_refactor.cpp src/kernel.cpp
trainRefactor_obj := $(trainRefactor_src:.cpp=.o)
trainRefactor_bin := train


# === Executable: train (training loop) ===
#train_src := src/train.cpp $(src_common)
#train_obj := $(train_src:.cpp=.o)
#train_bin := train

.PHONY: all clean run

#all: $(scribe_bin) $(record_bin) $(train_bin)

#$(scribe_bin): $(scribe_obj)
#	@echo "Linking GUI..."
#	$(CXX) $(CXXFLAGS) $^ $(libdir) $(lib) $(mac_frameworks) -o $@
#	@printf "$(green)Built DigitalScribe GUI$(black)\n"

$(trainRefactor_bin): $(trainRefactor_obj)
	@echo "Linking refactor..."
	$(CXX) $(CXXFLAGS) $^ $(libdir) $(lib) -o $@
	@printf "$(green)Built train executable$(black)\n"
	
$(record_bin): $(record_obj)
	@echo "Linking record_images..."
	$(CXX) $(CXXFLAGS) $^ $(libdir) $(lib) -o $@
	@printf "$(green)Built record_images tool$(black)\n"

#$(train_bin): $(train_obj)
#	@echo "Linking train..."
#	$(CXX) $(CXXFLAGS) $^ $(libdir) $(lib) -o $@
#	@printf "$(green)Built train executable$(black)\n"



%.o: %.cpp
	@printf "$(blue)Compiling $<...\n"
	@$(CXX) $(CXXFLAGS) -c $(inc) $(incl) $< -o $@

run: $(scribe_bin)
	./$(scribe_bin)

#clean:
#	rm -f $(scribe_bin) $(record_bin) $(train_bin) $(scribe_obj) $(record_obj) $(train_obj)
clean: 
	rm -f $(record_bin) $(trainrefactor_bin) $(trainRefactor_obj) $(record_obj)