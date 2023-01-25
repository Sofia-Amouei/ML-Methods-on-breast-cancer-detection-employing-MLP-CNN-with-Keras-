{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "BreastCancerDetection.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "from sklearn.datasets import load_breast_cancer\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense"
      ],
      "metadata": {
        "id": "wSMnN-PnCN5d"
      },
      "execution_count": 210,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data=load_breast_cancer()"
      ],
      "metadata": {
        "id": "ByBH4uz9CW0m"
      },
      "execution_count": 211,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.keys()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fXTDn6i8Cspt",
        "outputId": "b855ca1a-797b-454e-808e-e3729126fb26"
      },
      "execution_count": 212,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])"
            ]
          },
          "metadata": {},
          "execution_count": 212
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(data['DESCR'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "25PKNew1DoWB",
        "outputId": "d1cf9a42-2156-4472-b521-9ea6496ae4f7"
      },
      "execution_count": 213,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            ".. _breast_cancer_dataset:\n",
            "\n",
            "Breast cancer wisconsin (diagnostic) dataset\n",
            "--------------------------------------------\n",
            "\n",
            "**Data Set Characteristics:**\n",
            "\n",
            "    :Number of Instances: 569\n",
            "\n",
            "    :Number of Attributes: 30 numeric, predictive attributes and the class\n",
            "\n",
            "    :Attribute Information:\n",
            "        - radius (mean of distances from center to points on the perimeter)\n",
            "        - texture (standard deviation of gray-scale values)\n",
            "        - perimeter\n",
            "        - area\n",
            "        - smoothness (local variation in radius lengths)\n",
            "        - compactness (perimeter^2 / area - 1.0)\n",
            "        - concavity (severity of concave portions of the contour)\n",
            "        - concave points (number of concave portions of the contour)\n",
            "        - symmetry\n",
            "        - fractal dimension (\"coastline approximation\" - 1)\n",
            "\n",
            "        The mean, standard error, and \"worst\" or largest (mean of the three\n",
            "        worst/largest values) of these features were computed for each image,\n",
            "        resulting in 30 features.  For instance, field 0 is Mean Radius, field\n",
            "        10 is Radius SE, field 20 is Worst Radius.\n",
            "\n",
            "        - class:\n",
            "                - Malignant\n",
            "                - Benign\n",
            "\n",
            "    :Summary Statistics:\n",
            "\n",
            "    ===================================== ====== ======\n",
            "                                           Min    Max\n",
            "    ===================================== ====== ======\n",
            "    radius (mean):                        6.981  28.11\n",
            "    texture (mean):                       9.71   39.28\n",
            "    perimeter (mean):                     43.79  188.5\n",
            "    area (mean):                          143.5  2501.0\n",
            "    smoothness (mean):                    0.053  0.163\n",
            "    compactness (mean):                   0.019  0.345\n",
            "    concavity (mean):                     0.0    0.427\n",
            "    concave points (mean):                0.0    0.201\n",
            "    symmetry (mean):                      0.106  0.304\n",
            "    fractal dimension (mean):             0.05   0.097\n",
            "    radius (standard error):              0.112  2.873\n",
            "    texture (standard error):             0.36   4.885\n",
            "    perimeter (standard error):           0.757  21.98\n",
            "    area (standard error):                6.802  542.2\n",
            "    smoothness (standard error):          0.002  0.031\n",
            "    compactness (standard error):         0.002  0.135\n",
            "    concavity (standard error):           0.0    0.396\n",
            "    concave points (standard error):      0.0    0.053\n",
            "    symmetry (standard error):            0.008  0.079\n",
            "    fractal dimension (standard error):   0.001  0.03\n",
            "    radius (worst):                       7.93   36.04\n",
            "    texture (worst):                      12.02  49.54\n",
            "    perimeter (worst):                    50.41  251.2\n",
            "    area (worst):                         185.2  4254.0\n",
            "    smoothness (worst):                   0.071  0.223\n",
            "    compactness (worst):                  0.027  1.058\n",
            "    concavity (worst):                    0.0    1.252\n",
            "    concave points (worst):               0.0    0.291\n",
            "    symmetry (worst):                     0.156  0.664\n",
            "    fractal dimension (worst):            0.055  0.208\n",
            "    ===================================== ====== ======\n",
            "\n",
            "    :Missing Attribute Values: None\n",
            "\n",
            "    :Class Distribution: 212 - Malignant, 357 - Benign\n",
            "\n",
            "    :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian\n",
            "\n",
            "    :Donor: Nick Street\n",
            "\n",
            "    :Date: November, 1995\n",
            "\n",
            "This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.\n",
            "https://goo.gl/U2Uwz2\n",
            "\n",
            "Features are computed from a digitized image of a fine needle\n",
            "aspirate (FNA) of a breast mass.  They describe\n",
            "characteristics of the cell nuclei present in the image.\n",
            "\n",
            "Separating plane described above was obtained using\n",
            "Multisurface Method-Tree (MSM-T) [K. P. Bennett, \"Decision Tree\n",
            "Construction Via Linear Programming.\" Proceedings of the 4th\n",
            "Midwest Artificial Intelligence and Cognitive Science Society,\n",
            "pp. 97-101, 1992], a classification method which uses linear\n",
            "programming to construct a decision tree.  Relevant features\n",
            "were selected using an exhaustive search in the space of 1-4\n",
            "features and 1-3 separating planes.\n",
            "\n",
            "The actual linear program used to obtain the separating plane\n",
            "in the 3-dimensional space is that described in:\n",
            "[K. P. Bennett and O. L. Mangasarian: \"Robust Linear\n",
            "Programming Discrimination of Two Linearly Inseparable Sets\",\n",
            "Optimization Methods and Software 1, 1992, 23-34].\n",
            "\n",
            "This database is also available through the UW CS ftp server:\n",
            "\n",
            "ftp ftp.cs.wisc.edu\n",
            "cd math-prog/cpo-dataset/machine-learn/WDBC/\n",
            "\n",
            ".. topic:: References\n",
            "\n",
            "   - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction \n",
            "     for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on \n",
            "     Electronic Imaging: Science and Technology, volume 1905, pages 861-870,\n",
            "     San Jose, CA, 1993.\n",
            "   - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and \n",
            "     prognosis via linear programming. Operations Research, 43(4), pages 570-577, \n",
            "     July-August 1995.\n",
            "   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques\n",
            "     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) \n",
            "     163-171.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['data'].shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6CYuO3uAFC2H",
        "outputId": "73becc47-5f75-4c70-8285-1becf7675995"
      },
      "execution_count": 214,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(569, 30)"
            ]
          },
          "metadata": {},
          "execution_count": 214
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['feature_names']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "973UHwMOFNS2",
        "outputId": "53c93776-859b-4d11-a3fb-e3f4750b7095"
      },
      "execution_count": 215,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',\n",
              "       'mean smoothness', 'mean compactness', 'mean concavity',\n",
              "       'mean concave points', 'mean symmetry', 'mean fractal dimension',\n",
              "       'radius error', 'texture error', 'perimeter error', 'area error',\n",
              "       'smoothness error', 'compactness error', 'concavity error',\n",
              "       'concave points error', 'symmetry error',\n",
              "       'fractal dimension error', 'worst radius', 'worst texture',\n",
              "       'worst perimeter', 'worst area', 'worst smoothness',\n",
              "       'worst compactness', 'worst concavity', 'worst concave points',\n",
              "       'worst symmetry', 'worst fractal dimension'], dtype='<U23')"
            ]
          },
          "metadata": {},
          "execution_count": 215
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['data'][0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IwizKx3xFkaX",
        "outputId": "1ada0c55-b4f7-4975-b3ef-16a28ad39d2a"
      },
      "execution_count": 216,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,\n",
              "       3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,\n",
              "       8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,\n",
              "       3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,\n",
              "       1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01])"
            ]
          },
          "metadata": {},
          "execution_count": 216
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "j=0\n",
        "for i in data['feature_names']:\n",
        "  print(i,':',data['data'][0][j])\n",
        "  j+=1"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vJYsqYcwKAf5",
        "outputId": "b6de6977-632c-411b-dec1-ae76ec158cce"
      },
      "execution_count": 217,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mean radius : 17.99\n",
            "mean texture : 10.38\n",
            "mean perimeter : 122.8\n",
            "mean area : 1001.0\n",
            "mean smoothness : 0.1184\n",
            "mean compactness : 0.2776\n",
            "mean concavity : 0.3001\n",
            "mean concave points : 0.1471\n",
            "mean symmetry : 0.2419\n",
            "mean fractal dimension : 0.07871\n",
            "radius error : 1.095\n",
            "texture error : 0.9053\n",
            "perimeter error : 8.589\n",
            "area error : 153.4\n",
            "smoothness error : 0.006399\n",
            "compactness error : 0.04904\n",
            "concavity error : 0.05373\n",
            "concave points error : 0.01587\n",
            "symmetry error : 0.03003\n",
            "fractal dimension error : 0.006193\n",
            "worst radius : 25.38\n",
            "worst texture : 17.33\n",
            "worst perimeter : 184.6\n",
            "worst area : 2019.0\n",
            "worst smoothness : 0.1622\n",
            "worst compactness : 0.6656\n",
            "worst concavity : 0.7119\n",
            "worst concave points : 0.2654\n",
            "worst symmetry : 0.4601\n",
            "worst fractal dimension : 0.1189\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['target']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "30hpq-p-LG00",
        "outputId": "b8108020-9491-4e4d-da50-109238a97123"
      },
      "execution_count": 218,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,\n",
              "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,\n",
              "       1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,\n",
              "       1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,\n",
              "       1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,\n",
              "       0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,\n",
              "       1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,\n",
              "       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,\n",
              "       1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,\n",
              "       1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,\n",
              "       0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,\n",
              "       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,\n",
              "       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,\n",
              "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,\n",
              "       1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,\n",
              "       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,\n",
              "       1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,\n",
              "       1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])"
            ]
          },
          "metadata": {},
          "execution_count": 218
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['target_names']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cXfGdJ17Lhs8",
        "outputId": "35b77ffc-c1a6-4820-9f5e-4c449f6184c9"
      },
      "execution_count": 219,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['malignant', 'benign'], dtype='<U9')"
            ]
          },
          "metadata": {},
          "execution_count": 219
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['filename']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "UCJvX_UuMM7v",
        "outputId": "ff708ef5-03b4-4369-c5b1-9caad51e139f"
      },
      "execution_count": 220,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'breast_cancer.csv'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 220
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "feature=data['data']"
      ],
      "metadata": {
        "id": "IUA6EvULNTA4"
      },
      "execution_count": 221,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "label=data['target']"
      ],
      "metadata": {
        "id": "firyewN8NTT1"
      },
      "execution_count": 222,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "feature.shape\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cJ14NAw0NTXc",
        "outputId": "122d5d38-b97e-41ad-c4cb-f9c37ecab650"
      },
      "execution_count": 223,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(569, 30)"
            ]
          },
          "metadata": {},
          "execution_count": 223
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "label.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QZqdeURJNTbY",
        "outputId": "1c064d1b-3f75-41b1-b27c-2330be8304d8"
      },
      "execution_count": 224,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(569,)"
            ]
          },
          "metadata": {},
          "execution_count": 224
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "scale= StandardScaler()\n",
        "\n",
        "feature= scale.fit_transform(feature)"
      ],
      "metadata": {
        "id": "XO96Os6iNTik"
      },
      "execution_count": 225,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "j=0\n",
        "for i in data['feature_names']:\n",
        "  print(i,':',feature[0][j])\n",
        "  j+=1"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YA3PcEt_P76D",
        "outputId": "ec2780d8-0ce8-4d56-fd1a-b931dc351e9f"
      },
      "execution_count": 226,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "mean radius : 1.0970639814699807\n",
            "mean texture : -2.0733350146975935\n",
            "mean perimeter : 1.2699336881399383\n",
            "mean area : 0.9843749048031144\n",
            "mean smoothness : 1.568466329243428\n",
            "mean compactness : 3.2835146709868264\n",
            "mean concavity : 2.652873983743168\n",
            "mean concave points : 2.532475216403245\n",
            "mean symmetry : 2.2175150059646405\n",
            "mean fractal dimension : 2.255746885296269\n",
            "radius error : 2.4897339267376193\n",
            "texture error : -0.5652650590684639\n",
            "perimeter error : 2.833030865855184\n",
            "area error : 2.4875775569611043\n",
            "smoothness error : -0.21400164666895383\n",
            "compactness error : 1.3168615683959484\n",
            "concavity error : 0.72402615808036\n",
            "concave points error : 0.6608199414286064\n",
            "symmetry error : 1.1487566671861758\n",
            "fractal dimension error : 0.9070830809973359\n",
            "worst radius : 1.8866896251792757\n",
            "worst texture : -1.3592934737640827\n",
            "worst perimeter : 2.3036006236225606\n",
            "worst area : 2.0012374893299207\n",
            "worst smoothness : 1.3076862710715387\n",
            "worst compactness : 2.616665023512603\n",
            "worst concavity : 2.1095263465722556\n",
            "worst concave points : 2.296076127561788\n",
            "worst symmetry : 2.750622244124955\n",
            "worst fractal dimension : 1.9370146123781782\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(feature[0])\n",
        "print(data['target_names'][label[0]])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eNot0XxPP8LV",
        "outputId": "e33d4696-a4c1-4c37-80ea-328385575d64"
      },
      "execution_count": 227,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[ 1.09706398 -2.07333501  1.26993369  0.9843749   1.56846633  3.28351467\n",
            "  2.65287398  2.53247522  2.21751501  2.25574689  2.48973393 -0.56526506\n",
            "  2.83303087  2.48757756 -0.21400165  1.31686157  0.72402616  0.66081994\n",
            "  1.14875667  0.90708308  1.88668963 -1.35929347  2.30360062  2.00123749\n",
            "  1.30768627  2.61666502  2.10952635  2.29607613  2.75062224  1.93701461]\n",
            "malignant\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_frt=pd.DataFrame(feature, columns= data['feature_names'])\n",
        "df_lbl=pd.DataFrame(label, columns= ['label'])\n",
        "df=pd.concat([df_frt,df_lbl], axis=1)\n",
        "df= df.sample(frac=1)"
      ],
      "metadata": {
        "id": "DQBsQHciP8PD"
      },
      "execution_count": 228,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 523
        },
        "id": "bV-everRP8TE",
        "outputId": "7de257e6-bc5f-40c9-8658-e584c426d978"
      },
      "execution_count": 229,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
              "459    -1.241793      2.073499       -1.247611  -1.035222        -1.175673   \n",
              "253     0.901094     -0.514200        0.866270   0.777324         0.315955   \n",
              "351     0.460872     -0.016208        0.623248   0.294964         1.988342   \n",
              "45      1.284513     -0.393193        1.307005   1.197683         0.963560   \n",
              "250     1.934906      0.993739        1.933096   2.016784         0.308838   \n",
              "..           ...           ...             ...        ...              ...   \n",
              "112     0.037691      0.083856        0.241414  -0.071072        -1.280286   \n",
              "476     0.020650      0.288638        0.018164  -0.103779        -0.501736   \n",
              "548    -1.262242      0.011717       -1.273561  -1.050012        -0.814864   \n",
              "219     1.534446      3.067156        1.484123   1.615766        -0.865392   \n",
              "487     1.508885     -0.109290        1.488242   1.456496         0.892395   \n",
              "\n",
              "     mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
              "459         -1.100721       -0.921401            -0.992788      -0.695938   \n",
              "253         -0.004567        0.474586             0.892752       0.005043   \n",
              "351          2.502714        2.543646             1.941793       2.056873   \n",
              "45           1.217803        1.363478             1.340793       0.348232   \n",
              "250          1.066192        2.290035             2.117192       1.436213   \n",
              "..                ...             ...                  ...            ...   \n",
              "112          2.254449        2.655385             0.749595      -0.392910   \n",
              "476          0.122408       -0.479215            -0.473040      -1.115796   \n",
              "548         -1.024157       -0.821463            -1.013810      -0.845627   \n",
              "219          0.164101        0.322671             0.450127      -1.400570   \n",
              "487          0.766758        1.717529             1.817982       0.041553   \n",
              "\n",
              "     mean fractal dimension  ...  worst texture  worst perimeter  worst area  \\\n",
              "459               -0.464635  ...       1.830816        -1.168535   -0.932895   \n",
              "253               -0.945203  ...      -0.095626         0.704101    0.600181   \n",
              "351                1.875829  ...      -0.245442         0.361564    0.061029   \n",
              "45                -0.327128  ...      -0.709547         1.290882    1.206661   \n",
              "250               -0.541186  ...       0.215406         1.728734    1.985416   \n",
              "..                      ...  ...            ...              ...         ...   \n",
              "112                2.111151  ...      -0.317093        -0.007780   -0.301628   \n",
              "476               -0.383832  ...       0.257745         0.144127   -0.091558   \n",
              "548               -0.063453  ...      -0.014204        -1.136664   -0.907756   \n",
              "219               -1.370484  ...       3.213360         2.172543    2.806362   \n",
              "487               -0.233566  ...       0.767446         1.389175    1.510780   \n",
              "\n",
              "     worst smoothness  worst compactness  worst concavity  \\\n",
              "459         -0.936711          -0.912002        -0.960889   \n",
              "253          0.404667          -0.087565         0.314773   \n",
              "351          0.992068           1.592480         1.991028   \n",
              "45           1.557551           1.620470         2.217950   \n",
              "250         -0.493969           0.400354         2.048118   \n",
              "..                ...                ...              ...   \n",
              "112         -1.879621           1.049853         1.948330   \n",
              "476         -0.748217           0.563842        -0.100693   \n",
              "548         -0.546572          -1.010222        -0.857262   \n",
              "219          0.369598           0.988783         0.610780   \n",
              "487          0.834259           0.752140         1.541979   \n",
              "\n",
              "     worst concave points  worst symmetry  worst fractal dimension  label  \n",
              "459             -1.004137       -0.937918                -0.655891      1  \n",
              "253              1.082516        0.383809                -0.156041      0  \n",
              "351              1.505816        2.174692                 1.166735      0  \n",
              "45               1.875822        1.453162                 0.438017      0  \n",
              "250              1.460136        0.364396                -0.302339      0  \n",
              "..                    ...             ...                      ...    ...  \n",
              "112              0.546540       -0.813348                 1.344065      1  \n",
              "476              0.293779       -0.593330                -0.297351      1  \n",
              "548             -1.159448       -0.564210                -0.262993      1  \n",
              "219              0.729259       -0.303748                -0.458057      0  \n",
              "487              1.391616        0.590885                 0.340485      0  \n",
              "\n",
              "[569 rows x 31 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-11c2af7a-3251-408a-b2e5-fd758701077c\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>459</th>\n",
              "      <td>-1.241793</td>\n",
              "      <td>2.073499</td>\n",
              "      <td>-1.247611</td>\n",
              "      <td>-1.035222</td>\n",
              "      <td>-1.175673</td>\n",
              "      <td>-1.100721</td>\n",
              "      <td>-0.921401</td>\n",
              "      <td>-0.992788</td>\n",
              "      <td>-0.695938</td>\n",
              "      <td>-0.464635</td>\n",
              "      <td>...</td>\n",
              "      <td>1.830816</td>\n",
              "      <td>-1.168535</td>\n",
              "      <td>-0.932895</td>\n",
              "      <td>-0.936711</td>\n",
              "      <td>-0.912002</td>\n",
              "      <td>-0.960889</td>\n",
              "      <td>-1.004137</td>\n",
              "      <td>-0.937918</td>\n",
              "      <td>-0.655891</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>253</th>\n",
              "      <td>0.901094</td>\n",
              "      <td>-0.514200</td>\n",
              "      <td>0.866270</td>\n",
              "      <td>0.777324</td>\n",
              "      <td>0.315955</td>\n",
              "      <td>-0.004567</td>\n",
              "      <td>0.474586</td>\n",
              "      <td>0.892752</td>\n",
              "      <td>0.005043</td>\n",
              "      <td>-0.945203</td>\n",
              "      <td>...</td>\n",
              "      <td>-0.095626</td>\n",
              "      <td>0.704101</td>\n",
              "      <td>0.600181</td>\n",
              "      <td>0.404667</td>\n",
              "      <td>-0.087565</td>\n",
              "      <td>0.314773</td>\n",
              "      <td>1.082516</td>\n",
              "      <td>0.383809</td>\n",
              "      <td>-0.156041</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>351</th>\n",
              "      <td>0.460872</td>\n",
              "      <td>-0.016208</td>\n",
              "      <td>0.623248</td>\n",
              "      <td>0.294964</td>\n",
              "      <td>1.988342</td>\n",
              "      <td>2.502714</td>\n",
              "      <td>2.543646</td>\n",
              "      <td>1.941793</td>\n",
              "      <td>2.056873</td>\n",
              "      <td>1.875829</td>\n",
              "      <td>...</td>\n",
              "      <td>-0.245442</td>\n",
              "      <td>0.361564</td>\n",
              "      <td>0.061029</td>\n",
              "      <td>0.992068</td>\n",
              "      <td>1.592480</td>\n",
              "      <td>1.991028</td>\n",
              "      <td>1.505816</td>\n",
              "      <td>2.174692</td>\n",
              "      <td>1.166735</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>45</th>\n",
              "      <td>1.284513</td>\n",
              "      <td>-0.393193</td>\n",
              "      <td>1.307005</td>\n",
              "      <td>1.197683</td>\n",
              "      <td>0.963560</td>\n",
              "      <td>1.217803</td>\n",
              "      <td>1.363478</td>\n",
              "      <td>1.340793</td>\n",
              "      <td>0.348232</td>\n",
              "      <td>-0.327128</td>\n",
              "      <td>...</td>\n",
              "      <td>-0.709547</td>\n",
              "      <td>1.290882</td>\n",
              "      <td>1.206661</td>\n",
              "      <td>1.557551</td>\n",
              "      <td>1.620470</td>\n",
              "      <td>2.217950</td>\n",
              "      <td>1.875822</td>\n",
              "      <td>1.453162</td>\n",
              "      <td>0.438017</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>250</th>\n",
              "      <td>1.934906</td>\n",
              "      <td>0.993739</td>\n",
              "      <td>1.933096</td>\n",
              "      <td>2.016784</td>\n",
              "      <td>0.308838</td>\n",
              "      <td>1.066192</td>\n",
              "      <td>2.290035</td>\n",
              "      <td>2.117192</td>\n",
              "      <td>1.436213</td>\n",
              "      <td>-0.541186</td>\n",
              "      <td>...</td>\n",
              "      <td>0.215406</td>\n",
              "      <td>1.728734</td>\n",
              "      <td>1.985416</td>\n",
              "      <td>-0.493969</td>\n",
              "      <td>0.400354</td>\n",
              "      <td>2.048118</td>\n",
              "      <td>1.460136</td>\n",
              "      <td>0.364396</td>\n",
              "      <td>-0.302339</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>112</th>\n",
              "      <td>0.037691</td>\n",
              "      <td>0.083856</td>\n",
              "      <td>0.241414</td>\n",
              "      <td>-0.071072</td>\n",
              "      <td>-1.280286</td>\n",
              "      <td>2.254449</td>\n",
              "      <td>2.655385</td>\n",
              "      <td>0.749595</td>\n",
              "      <td>-0.392910</td>\n",
              "      <td>2.111151</td>\n",
              "      <td>...</td>\n",
              "      <td>-0.317093</td>\n",
              "      <td>-0.007780</td>\n",
              "      <td>-0.301628</td>\n",
              "      <td>-1.879621</td>\n",
              "      <td>1.049853</td>\n",
              "      <td>1.948330</td>\n",
              "      <td>0.546540</td>\n",
              "      <td>-0.813348</td>\n",
              "      <td>1.344065</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>476</th>\n",
              "      <td>0.020650</td>\n",
              "      <td>0.288638</td>\n",
              "      <td>0.018164</td>\n",
              "      <td>-0.103779</td>\n",
              "      <td>-0.501736</td>\n",
              "      <td>0.122408</td>\n",
              "      <td>-0.479215</td>\n",
              "      <td>-0.473040</td>\n",
              "      <td>-1.115796</td>\n",
              "      <td>-0.383832</td>\n",
              "      <td>...</td>\n",
              "      <td>0.257745</td>\n",
              "      <td>0.144127</td>\n",
              "      <td>-0.091558</td>\n",
              "      <td>-0.748217</td>\n",
              "      <td>0.563842</td>\n",
              "      <td>-0.100693</td>\n",
              "      <td>0.293779</td>\n",
              "      <td>-0.593330</td>\n",
              "      <td>-0.297351</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>548</th>\n",
              "      <td>-1.262242</td>\n",
              "      <td>0.011717</td>\n",
              "      <td>-1.273561</td>\n",
              "      <td>-1.050012</td>\n",
              "      <td>-0.814864</td>\n",
              "      <td>-1.024157</td>\n",
              "      <td>-0.821463</td>\n",
              "      <td>-1.013810</td>\n",
              "      <td>-0.845627</td>\n",
              "      <td>-0.063453</td>\n",
              "      <td>...</td>\n",
              "      <td>-0.014204</td>\n",
              "      <td>-1.136664</td>\n",
              "      <td>-0.907756</td>\n",
              "      <td>-0.546572</td>\n",
              "      <td>-1.010222</td>\n",
              "      <td>-0.857262</td>\n",
              "      <td>-1.159448</td>\n",
              "      <td>-0.564210</td>\n",
              "      <td>-0.262993</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>219</th>\n",
              "      <td>1.534446</td>\n",
              "      <td>3.067156</td>\n",
              "      <td>1.484123</td>\n",
              "      <td>1.615766</td>\n",
              "      <td>-0.865392</td>\n",
              "      <td>0.164101</td>\n",
              "      <td>0.322671</td>\n",
              "      <td>0.450127</td>\n",
              "      <td>-1.400570</td>\n",
              "      <td>-1.370484</td>\n",
              "      <td>...</td>\n",
              "      <td>3.213360</td>\n",
              "      <td>2.172543</td>\n",
              "      <td>2.806362</td>\n",
              "      <td>0.369598</td>\n",
              "      <td>0.988783</td>\n",
              "      <td>0.610780</td>\n",
              "      <td>0.729259</td>\n",
              "      <td>-0.303748</td>\n",
              "      <td>-0.458057</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>487</th>\n",
              "      <td>1.508885</td>\n",
              "      <td>-0.109290</td>\n",
              "      <td>1.488242</td>\n",
              "      <td>1.456496</td>\n",
              "      <td>0.892395</td>\n",
              "      <td>0.766758</td>\n",
              "      <td>1.717529</td>\n",
              "      <td>1.817982</td>\n",
              "      <td>0.041553</td>\n",
              "      <td>-0.233566</td>\n",
              "      <td>...</td>\n",
              "      <td>0.767446</td>\n",
              "      <td>1.389175</td>\n",
              "      <td>1.510780</td>\n",
              "      <td>0.834259</td>\n",
              "      <td>0.752140</td>\n",
              "      <td>1.541979</td>\n",
              "      <td>1.391616</td>\n",
              "      <td>0.590885</td>\n",
              "      <td>0.340485</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>569 rows × 31 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-11c2af7a-3251-408a-b2e5-fd758701077c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-11c2af7a-3251-408a-b2e5-fd758701077c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-11c2af7a-3251-408a-b2e5-fd758701077c');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 229
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#500 Training\n",
        "x_train = feature[:500]\n",
        "y_train = label[:500]\n",
        "\n",
        "#35 Validation\n",
        "x_val = feature[500:535]\n",
        "y_val = label[500:535]\n",
        "\n",
        "#34 Testing\n",
        "x_test = feature[535:]\n",
        "y_test = label[535:]"
      ],
      "metadata": {
        "id": "eYDJ5-9MP8Xz"
      },
      "execution_count": 230,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train.shape"
      ],
      "metadata": {
        "id": "Vmn-p32qP8dC",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "dcfdfe0f-3ea4-46b2-d823-0533e1759bd3"
      },
      "execution_count": 231,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(500, 30)"
            ]
          },
          "metadata": {},
          "execution_count": 231
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.backend import binary_crossentropy\n",
        "model=Sequential()\n",
        "model.add(Dense(64, activation='relu', input_dim= 30))\n",
        "model.add(Dense(32, activation='relu'))\n",
        "model.add(Dense(16, activation='relu'))\n",
        "model.add(Dense(1, activation='sigmoid'))\n",
        "\n",
        "model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'] )"
      ],
      "metadata": {
        "id": "FeoWq44EP8g8"
      },
      "execution_count": 232,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(x_train, y_train, batch_size=1, epochs=10, validation_data=(x_val, y_val))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ceoVIuRVGlby",
        "outputId": "6d686a55-694f-4200-8d3c-02ae6d89a909"
      },
      "execution_count": 233,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "500/500 [==============================] - 2s 2ms/step - loss: 0.1791 - accuracy: 0.9400 - val_loss: 0.0409 - val_accuracy: 1.0000\n",
            "Epoch 2/10\n",
            "500/500 [==============================] - 1s 2ms/step - loss: 0.0839 - accuracy: 0.9720 - val_loss: 0.0524 - val_accuracy: 0.9714\n",
            "Epoch 3/10\n",
            "500/500 [==============================] - 1s 2ms/step - loss: 0.0480 - accuracy: 0.9840 - val_loss: 0.0923 - val_accuracy: 0.9714\n",
            "Epoch 4/10\n",
            "500/500 [==============================] - 1s 2ms/step - loss: 0.0340 - accuracy: 0.9860 - val_loss: 0.1458 - val_accuracy: 0.9714\n",
            "Epoch 5/10\n",
            "500/500 [==============================] - 1s 2ms/step - loss: 0.0289 - accuracy: 0.9920 - val_loss: 0.1386 - val_accuracy: 0.9714\n",
            "Epoch 6/10\n",
            "500/500 [==============================] - 1s 2ms/step - loss: 0.0266 - accuracy: 0.9940 - val_loss: 0.1875 - val_accuracy: 0.9714\n",
            "Epoch 7/10\n",
            "500/500 [==============================] - 1s 2ms/step - loss: 0.0251 - accuracy: 0.9920 - val_loss: 0.1633 - val_accuracy: 0.9714\n",
            "Epoch 8/10\n",
            "500/500 [==============================] - 1s 2ms/step - loss: 0.0214 - accuracy: 0.9940 - val_loss: 0.1447 - val_accuracy: 0.9714\n",
            "Epoch 9/10\n",
            "500/500 [==============================] - 1s 2ms/step - loss: 0.0211 - accuracy: 0.9940 - val_loss: 0.1161 - val_accuracy: 0.9714\n",
            "Epoch 10/10\n",
            "500/500 [==============================] - 1s 2ms/step - loss: 0.0092 - accuracy: 0.9960 - val_loss: 0.1241 - val_accuracy: 0.9714\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7fcb16f48150>"
            ]
          },
          "metadata": {},
          "execution_count": 233
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x_train[0]"
      ],
      "metadata": {
        "id": "HETofOA6P8kx",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "746fcc4b-9d05-445b-b73e-37f2e111fc16"
      },
      "execution_count": 234,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([ 1.09706398, -2.07333501,  1.26993369,  0.9843749 ,  1.56846633,\n",
              "        3.28351467,  2.65287398,  2.53247522,  2.21751501,  2.25574689,\n",
              "        2.48973393, -0.56526506,  2.83303087,  2.48757756, -0.21400165,\n",
              "        1.31686157,  0.72402616,  0.66081994,  1.14875667,  0.90708308,\n",
              "        1.88668963, -1.35929347,  2.30360062,  2.00123749,  1.30768627,\n",
              "        2.61666502,  2.10952635,  2.29607613,  2.75062224,  1.93701461])"
            ]
          },
          "metadata": {},
          "execution_count": 234
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.evaluate(x_test, y_test)"
      ],
      "metadata": {
        "id": "NIr1PG-ZP8oV",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2d8ee904-f81e-4ec3-a49a-0ca1f602d8d1"
      },
      "execution_count": 235,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2/2 [==============================] - 0s 5ms/step - loss: 0.0105 - accuracy: 1.0000\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[0.01050428207963705, 1.0]"
            ]
          },
          "metadata": {},
          "execution_count": 235
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_test"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Dz0DKBJlJnP_",
        "outputId": "20284ad8-b3af-49ab-9100-b89c4827befd"
      },
      "execution_count": 236,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])"
            ]
          },
          "metadata": {},
          "execution_count": 236
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_val"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iag4cmc7Jnaa",
        "outputId": "74a85b99-4206-40f0-ce43-45321abdd5eb"
      },
      "execution_count": 237,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1])"
            ]
          },
          "metadata": {},
          "execution_count": 237
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sample= x_test[10]"
      ],
      "metadata": {
        "id": "FFcO7e5VL2X5"
      },
      "execution_count": 238,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample.shape"
      ],
      "metadata": {
        "id": "DxGZRo7fL2fU",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "edfba3d7-85d2-48cf-a523-c1a894d4ba1c"
      },
      "execution_count": 239,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(30,)"
            ]
          },
          "metadata": {},
          "execution_count": 239
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sample\n"
      ],
      "metadata": {
        "id": "RbfOtPesL2k-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e97efd91-9824-4e29-a7bc-15ac9b544b3d"
      },
      "execution_count": 240,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-0.14407806,  0.91694624, -0.19684934, -0.23233219, -0.27756523,\n",
              "       -0.6987597 , -0.741488  , -0.6316726 , -0.53894726, -0.67869351,\n",
              "       -0.21356438,  0.21617286, -0.39605381, -0.20015129, -0.39100882,\n",
              "       -0.25084001, -0.38739725, -0.44317904,  0.03967757, -0.45840426,\n",
              "       -0.19034815,  0.55574951, -0.28836304, -0.26506358, -0.47205092,\n",
              "       -0.65245698, -0.80257043, -0.65270672, -0.41860997, -0.7988643 ])"
            ]
          },
          "metadata": {},
          "execution_count": 240
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for i in range(30):\n",
        "  sample=x_test[i]\n",
        "  sample=np.reshape(sample,(1,30))\n",
        "\n",
        "  if model.predict(sample)[0][0] >0.5 :\n",
        "    print(\"-Benign\")\n",
        "  else:\n",
        "    print(\"-Malignant\")\n",
        "\n",
        "  if y_test[i]==1:\n",
        "    print(\"*Benign\")\n",
        "  else:\n",
        "    print(\"*Malignant\")  \n",
        "\n",
        "  print(\"------\")  "
      ],
      "metadata": {
        "id": "8K6MsDeCL2pR",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "45f69a46-7b08-4e91-fb67-14e549848fbc"
      },
      "execution_count": 241,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "-Malignant\n",
            "*Malignant\n",
            "------\n",
            "-Malignant\n",
            "*Malignant\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Benign\n",
            "*Benign\n",
            "------\n",
            "-Malignant\n",
            "*Malignant\n",
            "------\n",
            "-Malignant\n",
            "*Malignant\n",
            "------\n",
            "-Malignant\n",
            "*Malignant\n",
            "------\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sample.shape"
      ],
      "metadata": {
        "id": "13VevftxL2tJ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a3370a9b-f396-4e1c-8a43-536e815e50d2"
      },
      "execution_count": 242,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1, 30)"
            ]
          },
          "metadata": {},
          "execution_count": 242
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.predict(sample)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Mbw-pYYV5pNj",
        "outputId": "44a77287-65d6-4cad-8078-ead71833171a"
      },
      "execution_count": 243,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[4.8561056e-18]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 243
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x_test[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZYdz5VoX5kGD",
        "outputId": "8f9beb25-449b-4b46-e495-276f4a03eef5"
      },
      "execution_count": 244,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([ 1.82414032,  0.36543133,  1.88778669,  1.85751441,  0.58638323,\n",
              "        1.31824626,  1.50283863,  2.14814488,  1.15143953, -0.04077169,\n",
              "        1.05904307, -0.4114087 ,  0.910827  ,  1.04382537, -0.82102626,\n",
              "        0.03810891,  0.27043808,  0.3915519 , -0.12862155, -0.41830962,\n",
              "        1.66304049, -0.03211657,  1.57682617,  1.63207582, -0.24410428,\n",
              "        0.37681708,  0.82091153,  1.5256103 ,  0.28512459, -0.45750286])"
            ]
          },
          "metadata": {},
          "execution_count": 244
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_test[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6BSh_k365kMi",
        "outputId": "452c7374-7203-414a-e9fc-c96a8158757e"
      },
      "execution_count": 245,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0"
            ]
          },
          "metadata": {},
          "execution_count": 245
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x_test[10]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3UPeCUnc5kSJ",
        "outputId": "a1d28e1b-1125-4c98-925d-5b83297e9920"
      },
      "execution_count": 246,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([-0.14407806,  0.91694624, -0.19684934, -0.23233219, -0.27756523,\n",
              "       -0.6987597 , -0.741488  , -0.6316726 , -0.53894726, -0.67869351,\n",
              "       -0.21356438,  0.21617286, -0.39605381, -0.20015129, -0.39100882,\n",
              "       -0.25084001, -0.38739725, -0.44317904,  0.03967757, -0.45840426,\n",
              "       -0.19034815,  0.55574951, -0.28836304, -0.26506358, -0.47205092,\n",
              "       -0.65245698, -0.80257043, -0.65270672, -0.41860997, -0.7988643 ])"
            ]
          },
          "metadata": {},
          "execution_count": 246
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_test[10]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9L2_5-JO5kWk",
        "outputId": "b0c5603d-c058-49cb-d95d-b3688be21111"
      },
      "execution_count": 247,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1"
            ]
          },
          "metadata": {},
          "execution_count": 247
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "nWpeKHi-5kao"
      },
      "execution_count": 247,
      "outputs": []
    }
  ]
}
