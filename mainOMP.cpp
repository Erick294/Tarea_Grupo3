#include <iostream>
#include <vector>
#include "lodepng.h"
#include <fstream>
#include <omp.h>
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>

// Estructura para representar una imagen
struct Image {
    std::vector<unsigned char> pixels;
    unsigned width;
    unsigned height;
};

// Función para cargar una imagen PGM en una matriz de píxeles
bool loadPGM(const std::string &filename, Image &image) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_red), "Error al abrir el archivo: {}", filename);
        return false;
    }

    // Leer el encabezado de la imagen PGM
    std::string formato;
    file >> formato;
    file >> image.width >> image.height;

    int max_value;
    file >> max_value;

    // Verificar el tipo de formato PGM y convertir de P5 a P2 si es necesario
    if (formato == "P5") {
        // Leer los píxeles de la imagen en formato binario (P5)
        image.pixels.clear();
        image.pixels.resize(image.width * image.height);

        file.ignore(); // Ignorar el salto de línea después del valor máximo

        for (unsigned i = 0; i < image.width * image.height; i++) {
            unsigned char pixel_value;
            file.read(reinterpret_cast<char *>(&pixel_value), 1);
            image.pixels[i] = pixel_value;
        }
    } else if (formato == "P2") {
        // Leer los píxeles de la imagen en formato de texto (P2)
        image.pixels.clear();
        image.pixels.resize(image.width * image.height);

        for (unsigned i = 0; i < image.width * image.height; i++) {
            int pixel_value;
            file >> pixel_value;
            image.pixels[i] = static_cast<unsigned char>(pixel_value);
        }
    } else {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_red), "Formato de imagen no soportado: {} \n", formato);
        return false;
    }

    file.close();
    return true;
}


// Función para cargar una imagen PNG en una matriz de píxeles
bool loadPNG(const std::string &filename, Image &image) {
    std::vector<unsigned char> png_image; // Vector para almacenar los bytes de la imagen PNG
    unsigned error = lodepng::decode(png_image, image.width, image.height, filename);

    if (error) {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::dark_red), "Error al cargar la imagen: {} \n",
                   lodepng_error_text(error));
        return false;
    }

    // Convertir los bytes de la imagen a escala de grises (promedio de los componentes RGB)
    image.pixels.clear();
    image.pixels.resize(image.width * image.height);

    for (size_t i = 0; i < png_image.size(); i += 4) {
        unsigned char intensity = (png_image[i] + png_image[i + 1] + png_image[i + 2]) / 3;
        image.pixels[i / 4] = intensity;
    }

    return true;
}

// Función para crear el histograma de una imagen
std::vector<int> createHistogram(const Image &image) {
    std::vector<int> histogram(256, 0); // Inicializar el histograma con ceros

    for (const auto &pixel: image.pixels) {
        histogram[pixel]++;
    }

    return histogram;
}

// Función para crear el histograma de una imagen en escala de grises con bins
std::vector<int> createHistogramWithBins_serial(const Image &image, int num_bins) {
    std::vector<int> histogram(num_bins, 0);

    // Determinar el rango de intensidad de los píxeles (0 a 255)
    int intensity_min = 0;
    int intensity_max = 255;

    // Calcular el tamaño de cada bin
    int bin_size = std::ceil((intensity_max - intensity_min + 1) / static_cast<double>(num_bins));

    for (const auto &pixel: image.pixels) {
        int bin_index = (pixel - intensity_min) / bin_size;
        histogram[bin_index]++;
    }

    return histogram;
}

std::vector<int> createHistogramWithBins_omp(const Image &image, int num_bins) {
    std::vector<int> histogram(num_bins, 0);
    int intensity_min = 0;
    int intensity_max = 255;

    // Calcular el tamaño de cada bin
    int bin_size = (intensity_max - intensity_min + num_bins) / num_bins;

    std::vector<std::vector<int>> local_histograms(omp_get_max_threads(), std::vector<int>(num_bins, 0));

#pragma omp parallel for default(none) shared(image, intensity_min, bin_size, local_histograms)
    for (int i = 0; i < image.pixels.size(); i++) {
        int thread_id = omp_get_thread_num();
        int bin_index = (image.pixels[i] - intensity_min) / bin_size;
        local_histograms[thread_id][bin_index]++;
    }

#pragma omp critical
    // Combinar histogramas locales en el histograma final
    for (int t = 0; t < omp_get_max_threads(); t++) {
        for (int b = 0; b < num_bins; b++) {
            histogram[b] += local_histograms[t][b];
        }
    }

    return histogram;
}


int main() {
    std::string filenamePNG = "C:\\Users\\jkmoncayo\\Downloads\\GrupalHistograma\\img.png"; // Ruta de la imagen PNG
    std::string filenamePGM = "C:\\Users\\jkmoncayo\\Downloads\\GrupalHistograma\\ejem.pgm";
    Image imagePNG, imagePGM;

    if (!loadPNG(filenamePNG, imagePNG)) {
        return 1;
    }

    if (!loadPGM(filenamePGM, imagePGM)) {
        return 1;
    }

    int num_bins = 20; // Número de bins para el histograma

    //************************************************PNG*******************************************************************
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
               "********************************************************************* \n");
    fmt::print(fmt::emphasis::reverse | fg(fmt::color::gray), "Imagen PNG \n");
    {
        // Medir el tiempo para el código serial
        auto start_serial = std::chrono::high_resolution_clock::now();
        std::vector<int> histogram_bins = createHistogramWithBins_serial(imagePNG, num_bins);
        auto end_serial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_serial = end_serial - start_serial;

        // Medir el tiempo para el código con OpenMP
        auto start_omp = std::chrono::high_resolution_clock::now();
        std::vector<int> histogram_bins_omp = createHistogramWithBins_omp(imagePNG, num_bins);
        auto end_omp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_omp = end_omp - start_omp;

        // Calcular la suma total de los elementos en el histograma
        int total_sum = 0;
        for (int count: histogram_bins) {
            total_sum += count;
        }
        // Imprimir

        fmt::print(fmt::emphasis::bold | fg(fmt::color::aqua), "Tiempo serial: {} ms \n", duration_serial.count());
        fmt::print(fmt::emphasis::underline | fg(fmt::color::cornsilk), "Histograma serial:\n");
        for (int i = 0; i < num_bins; i++) {
            int bin_start = i * (256 / num_bins);
            int bin_end = (i == num_bins - 1) ? 255 : (i + 1) * (256 / num_bins) - 1;
            double porcentaje = static_cast<double>(histogram_bins[i]) / total_sum;
            fmt::print(fmt::emphasis::blink | fg(fmt::color::chartreuse), "{} - {}: {} ({:0.6} %)\n", bin_start,
                       bin_end,
                       histogram_bins[i], porcentaje * 100.0);
        }

        fmt::print(fmt::emphasis::bold | fg(fmt::color::beige),
                   "------------------------------------------------------------------\n");
        fmt::print(fmt::emphasis::bold | fg(fmt::color::beige),
                   "------------------------------------------------------------------\n");


        fmt::print(fmt::emphasis::bold | fg(fmt::color::aqua), "Tiempo omp: {} ms \n", duration_omp.count());
        fmt::print(fmt::emphasis::underline | fg(fmt::color::cornsilk), "Histograma omp:\n");
        for (int i = 0; i < num_bins; i++) {
            int bin_start = i * (256 / num_bins);
            int bin_end = (i == num_bins - 1) ? 255 : (i + 1) * (256 / num_bins) - 1;
            double porcentaje = static_cast<double>(histogram_bins_omp[i]) / total_sum;
            fmt::print(fmt::emphasis::blink | fg(fmt::color::chartreuse), "{} - {}: {} ({:0.6} %)\n", bin_start,
                       bin_end,
                       histogram_bins_omp[i], porcentaje * 100.0);
        }

    }

    //************************************************PGM*******************************************************************
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
               "********************************************************************* \n");
    fmt::print(fmt::emphasis::reverse | fg(fmt::color::gray), "Imagen PGM \n");
    {
        // Medir el tiempo para el código serial
        auto start_serial = std::chrono::high_resolution_clock::now();
        std::vector<int> histogram_bins = createHistogramWithBins_serial(imagePGM, num_bins);
        auto end_serial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_serial = end_serial - start_serial;

        // Medir el tiempo para el código con OpenMP
        auto start_omp = std::chrono::high_resolution_clock::now();
        std::vector<int> histogram_bins_omp = createHistogramWithBins_omp(imagePGM, num_bins);
        auto end_omp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_omp = end_omp - start_omp;

        // Calcular la suma total de los elementos en el histograma
        int total_sum = 0;
        for (int count: histogram_bins) {
            total_sum += count;
        }
        // Imprimir

        fmt::print(fmt::emphasis::bold | fg(fmt::color::aqua), "Tiempo serial: {} ms \n", duration_serial.count());
        fmt::print(fmt::emphasis::underline | fg(fmt::color::cornsilk), "Histograma serial:\n");
        for (int i = 0; i < num_bins; i++) {
            int bin_start = i * (256 / num_bins);
            int bin_end = (i == num_bins - 1) ? 255 : (i + 1) * (256 / num_bins) - 1;
            double porcentaje = static_cast<double>(histogram_bins[i]) / total_sum;
            fmt::print(fmt::emphasis::blink | fg(fmt::color::chartreuse), "{} - {}: {} ({:0.6} %)\n", bin_start,
                       bin_end,
                       histogram_bins[i], porcentaje * 100.0);
        }

        fmt::print(fmt::emphasis::bold | fg(fmt::color::beige),
                   "------------------------------------------------------------------\n");
        fmt::print(fmt::emphasis::bold | fg(fmt::color::beige),
                   "------------------------------------------------------------------\n");


        fmt::print(fmt::emphasis::bold | fg(fmt::color::aqua), "Tiempo omp: {} ms \n", duration_omp.count());
        fmt::print(fmt::emphasis::underline | fg(fmt::color::cornsilk), "Histograma omp:\n");
        for (int i = 0; i < num_bins; i++) {
            int bin_start = i * (256 / num_bins);
            int bin_end = (i == num_bins - 1) ? 255 : (i + 1) * (256 / num_bins) - 1;
            double porcentaje = static_cast<double>(histogram_bins_omp[i]) / total_sum;
            fmt::print(fmt::emphasis::blink | fg(fmt::color::chartreuse), "{} - {}: {} ({:0.6} %)\n", bin_start,
                       bin_end,
                       histogram_bins_omp[i], porcentaje * 100.0);
        }
    }


    return 0;
}