from __future__ import division

from modelos import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from prueba import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    #solicitamos los parametros para el entrenamiento, si no se especifica, la red neuronal se entrenará con los parametros por defectos que aqui se especifica
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=str, help="ruta en la que se encuentra el modelo para la  red neuronal")
    parser.add_argument("--peso_preentrenados", type=str, help="la ruta de el archivo donde estan los pesos de la red neuronal, si es la primera vez que se entrena se utiliza el de darknet de lo contrario los pesos de la ultima epoca realizada para entrenamiento")
    parser.add_argument("--configuracion_entrenamiento", type=str, help="ruta en la que se encuentra el archivo de configuracion (dicho archivo contiene el numero de clases a detectar, la ubicación de las rutass de las imagenes para entrenar y validar y la ruta donde se encuentra ubicado el archivo de los nombre de las clases)")
    parser.add_argument("--epochs", type=int, default=500, help="numero de epocas a realizar")
    parser.add_argument("--tamanio_lote", type=int, default=8, help="tamaño de cada lote de imagenes(enviadas simultaneamente para entrenar)")
    parser.add_argument("--n_cpu", type=int, default=8, help="numero de subprocesos de la CPU a utilizar durante la generación de lotes ")
    parametrosEntrenamiento = parser.parse_args()
    print(parametrosEntrenamiento)

    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #----------------------------------------------------------------------------
    #se recoge la configuracion para el entrenamiento, en estos parametros esta
    #la ruta del archivo donde se encuentra cada nombre de cada clase y la ruta de los archivo de entrenamiento
    #y validación que se crearon con generadorArchivoEntrenarValidar.py
    #---------------------------------------------------------------------------
    configuracion_entrenamiento = parse_data_config(parametrosEntrenamiento.configuracion_entrenamiento)
    train_path = configuracion_entrenamiento["entrenar"]
    valid_path = configuracion_entrenamiento["validos"]
    class_names = load_classes(configuracion_entrenamiento["nombres"])

    # se inicializa el modelo de darknet que previamente se solicita su ubicación
    model = Darknet(parametrosEntrenamiento.modelo).to(device)
    model.apply(weights_init_normal)

    # pesos preentrenamos desde una epoca terminan en .pth de lo contraro son los pesos  por defecto de darknet que se descargaron
    if parametrosEntrenamiento.peso_preentrenados:
        if parametrosEntrenamiento.peso_preentrenados.endswith(".pth"):
            model.load_state_dict(torch.load(parametrosEntrenamiento.peso_preentrenados))
        else:
            model.load_darknet_weights(parametrosEntrenamiento.peso_preentrenados)

    # se define cuales son los parametros que se necesitan en el entrenamiento con respecto a el numero de lotes que se va a enviar por iteración, numero de ccpu y los datos de la imagenes de entrenamiento
    dataset = ListDataset(train_path, augment=True, multiscale=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=parametrosEntrenamiento.tamanio_lote,
        shuffle=True,
        num_workers=parametrosEntrenamiento.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())
    # se define las metricas con las cuales se mostrará el estado actual del entrenamiento por iteración
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    #inicia la cuenta de epocas, dependiando cuantas definan entrara ese numero de veces en el for
    for epoch in range(parametrosEntrenamiento.epochs):
        #inicia el entrenamiento del modelo
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % 2:
                # Acumula gradiente antes de cada paso
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Progreso del registro
            # ----------------

            log_str = "\n---- [Epoca %d/%d, lote %d/%d] ----\n" % (epoch, parametrosEntrenamiento.epochs, batch_i, len(dataloader))

            metric_table = [["Metricas", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Carga las métricas en cada capa de YOLO, en total son 3
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # carga el Tensorboard y  muestra la perida que hubo en la iteracion
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("perdida(loss)", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)
            #añade la tabla con las metricas y la perdida total que hubo en la epoca
            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal perdida(loss) {loss.item()}"

            # Determine el tiempo aproximado que queda para la época (ETA)
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA(tiempo de finalizacion para la epoca) {time_left}"
            #imprime todo lo recolectado anteriormente
            print(log_str)

            model.seen += imgs.size(0)
        #Evaluar el modelo en el conjunto de validación
        if epoch % 1 == 1:
            print("\n---- Evaluación del modelo ----")
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=416,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # imprime la clase APs y el mAP
            ap_table = [["indice", "nombre de la clase", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
        #se define cada cuantas epocas se desea guardar un checkpoint del entrenamiento por default esta en casa epoca
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"pesosGenerados/punto_guardado_%d.pth" % epoch)
