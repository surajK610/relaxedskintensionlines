
import pandas as pd
import numpy as np
import glob

import cv2

import face_recognition
import dlib
from helpers import average, determine_side
from helpers import interpolate_pts_x, interpolate_pts_y, interpolate_pts


thickness = 1
color = (0, 255, 255)


def detect_single_face_dlib(img_rgb, rescale=(1, 1.5, 1, 1.2)):
    fd_front_dlib = dlib.get_frontal_face_detector()
    face = fd_front_dlib(img_rgb, 1)
    if len(face) > 0:
        face = sorted([(t.width() * t.height(), (t.left(), t.top(), t.width(), t.height()))
                       for t in face],
                      key=lambda t: t[0], reverse=True)[0][1]
    else:
        return None

    if rescale is not None and face is not None:
        if type(rescale) != tuple:
            rescale = (rescale, rescale, rescale, rescale)
        (x, y, w, h) = face

        w = min(img_rgb.shape[1] - x, int(w / 2 + rescale[2] * w / 2))
        h = min(img_rgb.shape[0] - y, int(h / 2 + rescale[3] * h / 2))

        fx = max(0, int(x + w / 2 * (1 - rescale[0])))
        fy = max(0, int(y + h / 2 * (1 - rescale[1])))
        fw = min(img_rgb.shape[1] - fx, int(w - w / 2 * (1 - rescale[0])))
        fh = min(img_rgb.shape[0] - fy, int(h - h / 2 * (1 - rescale[1])))

        face = (fx, fy, fw, fh)
    return face


# def draw_lines_frontal()


def draw_lines_frontal(im, draw_rectangle=False, draw_landmarks=True):
    face = detect_single_face_dlib(im)
    if draw_rectangle:
        im = cv2.rectangle(im, (face[0], face[1]), (face[0] + face[2] + 10, face[1] + face[3]), (0, 255, 0), 3)

    face_landmarks_list = face_recognition.face_landmarks(im)
    face_landmarks_list = face_landmarks_list[0]
    # print face_landmarks_list

    if draw_landmarks:
        for key, value in face_landmarks_list.iteritems():
            for point in value:
                im = cv2.circle(im, point, 2, [0, 0, 255], -3)

    nose_bridge = face_landmarks_list['nose_bridge']
    left_eye = face_landmarks_list['left_eye']
    nose_tip = face_landmarks_list['nose_tip']
    chin = face_landmarks_list['chin']
    right_eye = face_landmarks_list['right_eye']
    left_eyebrow = face_landmarks_list['left_eyebrow']
    bottom_lip = face_landmarks_list['bottom_lip']
    right_eyebrow = face_landmarks_list['right_eyebrow']
    top_lip = face_landmarks_list['top_lip']

    pts = np.array([chin[0], left_eyebrow[0], left_eyebrow[1], left_eyebrow[3],
                    np.average([left_eyebrow[4], right_eyebrow[0]], axis=0),
                    right_eyebrow[1], right_eyebrow[3], right_eyebrow[4], chin[16]], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([left_eye[3], nose_bridge[0], right_eye[0]], np.int32)
    pts = interpolate_pts_x(pts, kind='linear', range=int(pts[-1, 0] - pts[0, 0]) / 10)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt1 = np.average([np.average([left_eyebrow[4], right_eyebrow[0]], axis=0), np.array([nose_bridge[0]], np.float32)], axis=0)
    # in between the top of the nose bridge and the middle of the eyebrows
    pts = interpolate_pts_x(pts)
    pts = np.array([left_eye[2], ref_pt1[0], right_eye[1]], np.int32)

    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt2 = np.average([top_lip[2], nose_tip[1]], axis=0)
    ref_pt3 = np.average([top_lip[4], nose_tip[3]], axis=0)
    # close to center lip above
    pts = np.array([nose_tip[2], ref_pt2, top_lip[9]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([nose_tip[2], ref_pt3, top_lip[9]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # close to center lip above

    ref_pt4 = np.average([bottom_lip[4], chin[7]], axis=0)
    ref_pt5 = np.average([bottom_lip[2], chin[9]], axis=0)
    ref_pt6 = np.average([chin[7], chin[8]], axis=0)
    ref_pt7 = np.average([chin[8], chin[9]], axis=0)

    pts = np.array([top_lip[9], ref_pt4, ref_pt6], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([top_lip[9], ref_pt5, ref_pt7], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt8 = np.average([nose_tip[0], top_lip[0]], axis=0)
    ref_pt9 = np.average([nose_tip[4], top_lip[6]], axis=0)
    ref_pt10 = np.average([top_lip[10], top_lip[11]], axis=0)
    ref_pt11 = np.average([top_lip[7], top_lip[8]], axis=0)

    pts = np.array([nose_tip[0], ref_pt8, ref_pt10], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([nose_tip[4], ref_pt9, ref_pt11], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_tip[0], nose_tip[2], weights=[1, 2]), average(ref_pt2, ref_pt8, weights=[2, 1]),
                    average(ref_pt10, top_lip[9], weights=[1, 2])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_tip[2], nose_tip[4], weights=[2, 1]), average(ref_pt3, ref_pt9, weights=[2, 1]),
                    average(top_lip[9], ref_pt11, weights=[2, 1])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_tip[0], nose_tip[2], weights=[2, 1]), average(ref_pt2, ref_pt8, weights=[1, 2]),
                    average(ref_pt10, top_lip[9], weights=[2, 1])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_tip[2], nose_tip[4], weights=[1, 2]), average(ref_pt3, ref_pt9, weights=[1, 2]),
                    average(top_lip[9], ref_pt11, weights=[1, 2])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt12 = np.average([chin[6], bottom_lip[5]], axis=0)
    ref_pt13 = np.average([chin[10], bottom_lip[1]], axis=0)
    ref_pt14 = np.average([chin[6], chin[7]], axis=0)
    ref_pt15 = np.average([chin[9], chin[10]], axis=0)

    pts = np.array([ref_pt10, ref_pt12, ref_pt14], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt11, ref_pt13, ref_pt15], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt10, top_lip[9], weights=[1, 2]), average(ref_pt4, ref_pt12, weights=[2, 1]),
                    average(ref_pt6, ref_pt14, weights=[2, 1])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(top_lip[9], ref_pt11, weights=[2, 1]), average(ref_pt5, ref_pt13, weights=[2, 1]),
                    average(ref_pt7, ref_pt15, weights=[2, 1])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt10, top_lip[9], weights=[2, 1]), average(ref_pt4, ref_pt12, weights=[1, 2]),
                    average(ref_pt6, ref_pt14, weights=[1, 2])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(top_lip[9], ref_pt11, weights=[1, 2]), average(ref_pt5, ref_pt13, weights=[1, 2]),
                    average(ref_pt7, ref_pt15, weights=[1, 2])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([bottom_lip[3], average(ref_pt4, ref_pt5, weights=[2.5, 1]), chin[8]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([bottom_lip[3], average(ref_pt4, ref_pt5, weights=[1, 2.5]), chin[8]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt16 = np.average([chin[6], top_lip[0]], weights=[1, 1], axis=0)
    ref_pt17 = np.average([chin[10], top_lip[6]], weights=[1, 1], axis=0)
    ref_pt18 = np.average([chin[5], chin[6]], axis=0)
    ref_pt19 = np.average([chin[10], chin[11]], axis=0)

    pts = np.array([top_lip[11], ref_pt16, chin[6]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([top_lip[7], ref_pt17, chin[10]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # pts = np.array([top_lip[11], average(top_lip[11], nose_bridge[1], weights=[2, 1]), nose_tip[0]], np.int32)
    # pts = interpolate_pts_y(pts)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)
    #
    # pts = np.array([top_lip[7], nose_tip[4]], np.int32)
    # pts = interpolate_pts_y(pts)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt20 = np.average([chin[3], nose_bridge[2]], weights=[1, 1.7], axis=0)
    ref_pt21 = np.average([chin[13], nose_bridge[2]], weights=[1, 1.75], axis=0)
    # ref_pt22 = np.average([chin[4], nose_bridge[2]], weights=[1, 1], axis=0)
    # ref_pt23 = np.average([chin[12], nose_bridge[2]], weights=[1, 1], axis=0)

    ref_pt22 = np.average([chin[3], nose_tip[0]], weights=[1, 1.5], axis=0)
    ref_pt23 = np.average([chin[13], nose_tip[4]], weights=[1, 1.5], axis=0)

    # pts = np.array([nose_bridge[2], ref_pt20, ref_pt22, top_lip[0]], np.int32)
    # pts = interpolate_pts_y(pts, on=False)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)
    #
    # pts = np.array([nose_bridge[2], ref_pt21, ref_pt23, top_lip[6]], np.int32)
    # pts = interpolate_pts_y(pts, on=False)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([nose_bridge[2], ref_pt20, ref_pt22, ref_pt18], np.int32)
    pts = interpolate_pts_y(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([nose_bridge[2], ref_pt21, ref_pt23, ref_pt19], np.int32)
    pts = interpolate_pts_y(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_bridge[2], nose_bridge[3]), average(ref_pt20, nose_tip[0]),
                    average(ref_pt22, top_lip[0]), average(ref_pt18, chin[6])], np.int32)
    pts = interpolate_pts_y(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_bridge[2], nose_bridge[3]), average(ref_pt21, nose_tip[4]),
                    average(ref_pt23, top_lip[6]), average(ref_pt19, chin[10])], np.int32)
    pts = interpolate_pts_y(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt24 = np.average([nose_bridge[1], nose_bridge[2]], weights=[1, 1.5], axis=0)
    ref_pt25 = np.average([chin[2], nose_bridge[2]], weights=[1, 1.5], axis=0)
    ref_pt26 = np.average([chin[14], nose_bridge[2]], weights=[1, 1.5], axis=0)
    ref_pt27 = np.average([chin[3], nose_bridge[2]], weights=[1.6, 1], axis=0)
    ref_pt28 = np.average([chin[13], nose_bridge[2]], weights=[1.6, 1], axis=0)
    ref_pt29 = np.average([chin[4], chin[5]], weights=[1, 1.5], axis=0)
    ref_pt30 = np.average([chin[11], chin[12]], weights=[1.5, 1], axis=0)

    # pts = np.array([ref_pt24, ref_pt25, ref_pt27, ref_pt29], np.int32)
    pts = np.array([ref_pt29, ref_pt27, ref_pt25, ref_pt24], np.int32)
    # pts = interpolate_pts_x(pts, on=True, kind='linear', range=20)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt24, ref_pt26, ref_pt28, ref_pt30], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([chin[5], average(ref_pt27, ref_pt22), average(ref_pt25, ref_pt20, weights=[1, 2]),
                    average(ref_pt24, nose_bridge[2])], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(ref_pt24, nose_bridge[2]), average(ref_pt26, ref_pt21, weights=[1, 2]), average(ref_pt28, ref_pt23),
         chin[11]], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # pts= np.array([chin[5], average(nose_bridge[2], chin[4], weights=[1.5, 1]), nose_bridge[2] ], np.int32)
    # pts = interpolate_pts(pts, on=True)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # pts= np.array([chin[11], average(nose_bridge[2], chin[13], weights=[1.5, 1]), nose_bridge[2]  ], np.int32)
    # pts = interpolate_pts(pts, on=True)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt31 = np.average([nose_bridge[1], nose_bridge[2]], weights=[3, 1], axis=0)
    ref_pt32 = np.average([chin[2], nose_bridge[1]], weights=[1, 1.15], axis=0)
    ref_pt33 = np.average([chin[14], nose_bridge[1]], weights=[1, 1.15], axis=0)
    ref_pt34 = np.average([chin[3], nose_bridge[0]], weights=[4, 1], axis=0)
    ref_pt35 = np.average([chin[13], nose_bridge[0]], weights=[4, 1], axis=0)
    ref_pt36 = np.average([chin[3], chin[4]], weights=[1, 1.5], axis=0)
    ref_pt37 = np.average([chin[12], chin[13]], weights=[1.5, 1], axis=0)

    pts = np.array([ref_pt36, ref_pt34, ref_pt32, ref_pt31], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt31, ref_pt33, ref_pt35, ref_pt37], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([chin[4], average(ref_pt34, ref_pt27), average(ref_pt32, ref_pt25), average(ref_pt31, ref_pt24)],
                   np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt31, ref_pt24), average(ref_pt33, ref_pt26), average(ref_pt35, ref_pt28), chin[12]],
                   np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt38 = np.average([nose_bridge[0], nose_bridge[1]], weights=[1, 2], axis=0)
    ref_pt39 = np.average([chin[2], nose_bridge[0]], weights=[1, 1.15], axis=0)
    ref_pt40 = np.average([chin[14], nose_bridge[0]], weights=[1, 1.15], axis=0)
    ref_pt41 = np.average([chin[2], nose_bridge[1]], weights=[4, 1], axis=0)
    ref_pt42 = np.average([chin[14], nose_bridge[1]], weights=[4, 1], axis=0)
    ref_pt43 = np.average([chin[2], chin[3]], weights=[1, 2], axis=0)
    ref_pt44 = np.average([chin[13], chin[14]], weights=[2, 1], axis=0)

    pts = np.array([ref_pt43, ref_pt41, ref_pt39, ref_pt38], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt38, ref_pt40, ref_pt42, ref_pt44], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([chin[3], average(ref_pt41, ref_pt34), average(ref_pt39, ref_pt32), average(ref_pt38, ref_pt31)],
                   np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt38, ref_pt31), average(ref_pt40, ref_pt33), average(ref_pt42, ref_pt35), chin[13]],
                   np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt45 = np.average([nose_bridge[0], nose_bridge[1]], weights=[2, 1], axis=0)
    ref_pt46 = np.average([left_eye[3], nose_bridge[1]], weights=[1, 1], axis=0)
    ref_pt47 = np.average([right_eye[0], nose_bridge[1]], weights=[1, 1], axis=0)
    ref_pt48 = np.average([(np.average([chin[2], chin[1]], axis=0)), nose_bridge[0]], weights=[1, 1.5], axis=0)
    ref_pt49 = np.average([(np.average([chin[14], chin[15]], axis=0)), nose_bridge[0]], weights=[1, 1.5], axis=0)
    ref_pt50 = np.average([chin[2], left_eye[5]], weights=[1, 2], axis=0)
    ref_pt51 = np.average([chin[14], right_eye[4]], weights=[1, 2], axis=0)
    ref_pt52 = np.average([chin[1], chin[2]], weights=[1, 2], axis=0)
    ref_pt53 = np.average([chin[14], chin[15]], weights=[2, 1], axis=0)

    pts = np.array([ref_pt52, average(average(chin[1], chin[2], weights=[4, 1]), left_eye[5]),  # ref_pt50,
                    ref_pt48, ref_pt46, ref_pt45], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt45, ref_pt47, ref_pt49,  # ref_pt51,
                    average(average(chin[15], chin[14], weights=[4, 1]), right_eye[4]), ref_pt53], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt52, ref_pt43), average(average(average(chin[1], chin[2], weights=[4, 1]),
                                                                 left_eye[5]), ref_pt41), average(ref_pt39, ref_pt48)],
                   np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt49, ref_pt40), average(average(average(chin[15], chin[14], weights=[4, 1]),
                                                                 right_eye[4]), ref_pt42), average(ref_pt53, ref_pt44)],
                   np.int32)
    # pts = interpolate_pts_x(pts, kind='linear', range=int(pts[-1, 0] - pts[0, 0]) / 10)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # ref_pt54 = np.average([chin[1], left_eye[0]], weights=[1, 1.15], axis=0)
    # ref_pt55 = np.average([chin[15], right_eye[3]], weights=[1, 1.15], axis=0)

    # pts = np.array([average(chin[1], chin[2], weights=[1, 2]), ref_pt54, left_eye[5]], np.int32)
    # pts = interpolate_pts_x(pts, on=True)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)
    #
    # pts = np.array([right_eye[4], ref_pt55,  average(chin[15], chin[14], weights=[1, 2])], np.int32)
    # pts = interpolate_pts_x(pts, on=True)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)


    ref_pt58 = np.average([chin[0], chin[1]], weights=[2, 1], axis=0)
    ref_pt59 = np.average([chin[16], chin[15]], weights=[2, 1], axis=0)

    ref_pt56 = np.average([ref_pt58, left_eye[0]], weights=[1, 1.5], axis=0)
    ref_pt57 = np.average([ref_pt59, right_eye[3]], weights=[1, 1.5], axis=0)

    pts = np.array([average(chin[0], chin[1], weights=[1, 1]), ref_pt56, left_eye[0]], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], ref_pt57, average(chin[15], chin[16], weights=[1, 1])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([chin[0], ref_pt56, left_eye[0]], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], ref_pt57, chin[16]], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt60 = np.average([chin[0], chin[1]], weights=[1, 2], axis=0)
    ref_pt61 = np.average([chin[16], chin[15]], weights=[1, 2], axis=0)

    ref_pt62 = np.average([ref_pt60, left_eye[0]], weights=[1, 1.5], axis=0)
    ref_pt63 = np.average([ref_pt61, right_eye[3]], weights=[1, 1.5], axis=0)

    pts = np.array([chin[1], ref_pt62, left_eye[0]], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], ref_pt63, chin[15]], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[1], chin[2]), ref_pt62, left_eye[0]], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], ref_pt63, average(chin[15], chin[14])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[0], left_eyebrow[0]), average(chin[0], left_eye[0], weights=[1, 3]), left_eye[0]],
                   np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], average(chin[16], right_eye[3], weights=[1, 3]), average(chin[16], right_eyebrow[4])],
                   np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # circle eye middle
    pts = np.array([average(chin[0], left_eye[0], weights=[1, 3]), average(left_eyebrow[2], left_eye[1]),
                    average(left_eyebrow[3], left_eye[1]), average(left_eye[3], right_eye[0], weights=[4, 1])],
                   np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[0], left_eye[0], weights=[1, 3]), average(average(ref_pt39, ref_pt48), left_eye[5]),
                    average(average(ref_pt39, ref_pt48), left_eye[4]),
                    average(left_eye[3], right_eye[0], weights=[4, 1])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 4]), average(right_eyebrow[1], right_eye[1], weights=[1, 1]),
         average(right_eyebrow[2], right_eye[2]), average(chin[16], right_eye[3], weights=[1, 3])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 4]), average(average(ref_pt49, ref_pt40), right_eye[5]),
         average(average(ref_pt49, ref_pt40), right_eye[4]), average(chin[16], right_eye[3], weights=[1, 3])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # circle eye close
    pts = np.array(
        [average(chin[0], left_eye[0], weights=[1, 8]), average(left_eyebrow[2], left_eye[1], weights=[1, 3]),
         average(left_eyebrow[3], left_eye[1], weights=[1, 3]), average(left_eye[3], right_eye[0], weights=[8, 1])],
        np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[0], left_eye[0], weights=[1, 8]),
                    average(average(ref_pt39, ref_pt48), left_eye[5], weights=[1, 4]),
                    average(average(ref_pt39, ref_pt48), left_eye[4], weights=[1, 4]),
                    average(left_eye[3], right_eye[0], weights=[6, 1])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 8]), average(right_eyebrow[1], right_eye[1], weights=[1, 4]),
         average(right_eyebrow[2], right_eye[2], weights=[1, 4]), average(chin[16], right_eye[3], weights=[1, 8])],
        np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 8]),
         average(average(ref_pt49, ref_pt40), right_eye[5], weights=[1, 3]),
         average(average(ref_pt49, ref_pt40), right_eye[4], weights=[1, 3]),
         average(chin[16], right_eye[3], weights=[1, 8])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # circle eye far
    pts = np.array(
        [average(chin[0], left_eye[0], weights=[1, 1.25]), average(left_eyebrow[2], left_eye[1], weights=[3, 1]),
         average(left_eyebrow[3], left_eye[1], weights=[3, 1]), average(left_eye[3], right_eye[0], weights=[2, 1])],
        np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[0], left_eye[0], weights=[1, 1.25]),
                    average(ref_pt39, left_eye[5], weights=[4, 1]),
                    average(ref_pt39, left_eye[4], weights=[4, 1]),
                    average(left_eye[3], right_eye[0], weights=[2, 1])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 2]), average(right_eyebrow[1], right_eye[1], weights=[3, 1]),
         average(right_eyebrow[2], right_eye[2], weights=[3, 1]), average(chin[16], right_eye[3], weights=[1, 1.25])],
        np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 2]), average(ref_pt40, right_eye[5], weights=[4, 1]),
         average(ref_pt40, right_eye[4], weights=[4, 1]), average(chin[16], right_eye[3], weights=[1, 1.25])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    boundary_upper = face[1]
    boundary_lower = left_eyebrow[3]
    boundary_lower = boundary_lower[1]
    diff = boundary_lower - boundary_upper

    # chin2 = chin[2]
    # chin2 = chin2[0]
    # print chin2
    # chin14 = chin[14]
    # chin14 = chin14[0]
    #
    # chin0 = chin[0]
    # print chin0[0]
    # chin0 = chin0[1]
    #
    # chin16 = chin[16]
    # chin16 = chin16[1]

    # pts_lower = np.array([(chin2, chin0), left_eyebrow[0], left_eyebrow[1], left_eyebrow[3],
    #                       right_eyebrow[1], right_eyebrow[3], right_eyebrow[4], (chin14, chin16)], np.int32)

    pts_lower = np.array([left_eyebrow[0], left_eyebrow[1], left_eyebrow[3],
                          right_eyebrow[1], right_eyebrow[3], right_eyebrow[4]], np.int32)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 5 * 2
    pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)
    pts = interpolate_pts_x(pts)
    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 5 * 1
    pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = interpolate_pts_x(pts)
    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 5 * 3
    pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = interpolate_pts_x(pts)
    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 4 * 3
    pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = interpolate_pts_x(pts)
    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 10 * 3
    pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = interpolate_pts_x(pts)
    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 2 * 1
    pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = interpolate_pts_x(pts)
    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 10 * 1
    pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = interpolate_pts_x(pts)
    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    return im


def draw_lines_semifrontal(im, draw_rectangle=False, draw_landmarks=True):
    face = detect_single_face_dlib(im)
    if draw_rectangle:
        im = cv2.rectangle(im, (face[0], face[1]), (face[0] + face[2] + 10, face[1] + face[3]), (0, 255, 0), 3)

    face_landmarks_list = face_recognition.face_landmarks(im)
    face_landmarks_list = face_landmarks_list[0]
    # print face_landmarks_list

    if draw_landmarks:
        for key, value in face_landmarks_list.iteritems():
            for point in value:
                im = cv2.circle(im, point, 2, [0, 0, 255], -3)

    nose_bridge = face_landmarks_list['nose_bridge']
    left_eye = face_landmarks_list['left_eye']
    nose_tip = face_landmarks_list['nose_tip']
    chin = face_landmarks_list['chin']
    right_eye = face_landmarks_list['right_eye']
    left_eyebrow = face_landmarks_list['left_eyebrow']
    bottom_lip = face_landmarks_list['bottom_lip']
    right_eyebrow = face_landmarks_list['right_eyebrow']
    top_lip = face_landmarks_list['top_lip']

    side = determine_side(chin, right_eye, left_eye)
    # pts = np.array([chin[0], left_eyebrow[0], left_eyebrow[1], left_eyebrow[2], left_eyebrow[3], left_eyebrow[4],
    #                right_eyebrow[0], right_eyebrow[1], right_eyebrow[2], right_eyebrow[3], right_eyebrow[4], chin[-1]], np.int32)
    pts = np.array([chin[0], left_eyebrow[0], left_eyebrow[1], left_eyebrow[3], np.average([left_eyebrow[4], right_eyebrow[0]], axis=0),
                    right_eyebrow[1], right_eyebrow[3], right_eyebrow[4], chin[16]], np.int32)
    pts = interpolate_pts_x(pts, on=False)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([left_eye[3], nose_bridge[0], right_eye[0]], np.int32)
    pts = interpolate_pts_x(pts, kind='linear', range= int(pts[-1, 0] - pts[0, 0])/10)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt1 = np.average(
        [np.average([left_eyebrow[4], right_eyebrow[0]], axis=0), np.array([nose_bridge[0]], np.float32)], axis=0)
    # in between the top of the nose bridge and the middle of the eyebrows
    pts = interpolate_pts_x(pts)
    pts = np.array([left_eye[2], ref_pt1[0], right_eye[1]], np.int32)

    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt2 = np.average([top_lip[2], nose_tip[1]], axis=0)
    ref_pt3 = np.average([top_lip[4], nose_tip[3]], axis=0)
    # close to center lip above
    pts = np.array([nose_tip[2], ref_pt2, top_lip[9]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([nose_tip[2], ref_pt3, top_lip[9]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # close to center lip above

    ref_pt4 = np.average([bottom_lip[4], chin[7]], axis=0)
    ref_pt5 = np.average([bottom_lip[2], chin[9]], axis=0)
    ref_pt6 = np.average([chin[7], chin[8]], axis=0)
    ref_pt7 = np.average([chin[8], chin[9]], axis=0)

    pts = np.array([top_lip[9], ref_pt4, ref_pt6], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([top_lip[9], ref_pt5, ref_pt7], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt8 = np.average([nose_tip[0], top_lip[0]], axis=0)
    ref_pt9 = np.average([nose_tip[4], top_lip[6]], axis=0)
    ref_pt10 = np.average([top_lip[10], top_lip[11]], axis=0)
    ref_pt11 = np.average([top_lip[7], top_lip[8]], axis=0)

    pts = np.array([nose_tip[0], ref_pt8, ref_pt10], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([nose_tip[4], ref_pt9, ref_pt11], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_tip[0], nose_tip[2], weights=[1, 2]), average(ref_pt2, ref_pt8, weights=[2, 1]), average(ref_pt10, top_lip[9], weights=[1, 2])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_tip[2], nose_tip[4], weights=[2, 1]), average(ref_pt3, ref_pt9, weights=[2, 1]), average(top_lip[9], ref_pt11, weights=[2, 1])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_tip[0], nose_tip[2], weights=[2, 1]), average(ref_pt2, ref_pt8, weights=[1, 2]), average(ref_pt10, top_lip[9], weights=[2, 1])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(nose_tip[2], nose_tip[4], weights=[1, 2]), average(ref_pt3, ref_pt9, weights=[1, 2]), average(top_lip[9], ref_pt11, weights=[1, 2])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt12 = np.average([chin[6], bottom_lip[5]], axis=0)
    ref_pt13 = np.average([chin[10], bottom_lip[1]], axis=0)
    ref_pt14 = np.average([chin[6], chin[7]], axis=0)
    ref_pt15 = np.average([chin[9], chin[10]], axis=0)

    pts = np.array([ref_pt10, ref_pt12, ref_pt14], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt11, ref_pt13, ref_pt15], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt10, top_lip[9], weights=[1, 2]), average(ref_pt4, ref_pt12, weights=[2, 1]),
                    average(ref_pt6, ref_pt14, weights=[2, 1])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(top_lip[9], ref_pt11, weights=[2, 1]), average(ref_pt5, ref_pt13, weights=[2, 1]),
                    average(ref_pt7, ref_pt15, weights=[2, 1])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt10, top_lip[9], weights=[2, 1]), average(ref_pt4, ref_pt12, weights=[1, 2]),
                    average(ref_pt6, ref_pt14, weights=[1, 2])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(top_lip[9], ref_pt11, weights=[1, 2]), average(ref_pt5, ref_pt13, weights=[1, 2]),
                    average(ref_pt7, ref_pt15, weights=[1, 2])], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([bottom_lip[3], average(ref_pt4, ref_pt5, weights=[2.5, 1]), chin[8]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([bottom_lip[3], average(ref_pt4, ref_pt5, weights=[1, 2.5]), chin[8]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)


    ref_pt16 = np.average([chin[6], top_lip[0]], weights=[1, 1], axis=0)
    ref_pt17 = np.average([chin[10], top_lip[6]], weights=[1, 1],  axis=0)
    ref_pt18 = np.average([chin[5], chin[6]], axis=0)
    ref_pt19 = np.average([chin[10], chin[11]], axis=0)

    pts = np.array([top_lip[11], ref_pt16, chin[6]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([top_lip[7], ref_pt17, chin[10]], np.int32)
    pts = interpolate_pts_y(pts)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # pts = np.array([top_lip[11], average(top_lip[11], nose_bridge[1], weights=[2, 1]), nose_tip[0]], np.int32)
    # pts = interpolate_pts_y(pts)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)
    #
    # pts = np.array([top_lip[7], nose_tip[4]], np.int32)
    # pts = interpolate_pts_y(pts)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt20 = np.average([chin[3], nose_bridge[2]], weights=[1, 1.75], axis=0)
    ref_pt21 = np.average([chin[13], nose_bridge[2]], weights=[1, 1.75], axis=0)
    # ref_pt22 = np.average([chin[4], nose_bridge[2]], weights=[1, 1], axis=0)
    # ref_pt23 = np.average([chin[12], nose_bridge[2]], weights=[1, 1], axis=0)

    ref_pt22 = np.average([chin[3], nose_tip[0]], weights=[1, 1.5], axis=0)
    ref_pt23 = np.average([chin[13], nose_tip[4]], weights=[1, 1.5], axis=0)

    # pts = np.array([nose_bridge[2], ref_pt20, ref_pt22, ref_pt18], np.int32)
    # pts = interpolate_pts_y(pts, on=False)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)
    #
    # pts = np.array([nose_bridge[2], ref_pt21, ref_pt23, ref_pt19], np.int32)
    # pts = interpolate_pts_y(pts, on=False)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt18, ref_pt22, ref_pt20, nose_bridge[2]], np.int32)
    pts = interpolate_pts_x(pts, on=False)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([nose_bridge[2], ref_pt21, ref_pt23, ref_pt19], np.int32)
    pts = interpolate_pts_y(pts, on=False)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    # pts = np.array([average(nose_bridge[2], nose_bridge[3]), average(ref_pt20, nose_tip[0]),
    #                 average(ref_pt22, top_lip[0]), average(ref_pt18, chin[6])], np.int32)
    # pts = interpolate_pts_y(pts, on=False)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)
    #
    # pts = np.array([average(nose_bridge[2], nose_bridge[3]), average(ref_pt21, nose_tip[4]),
    #                 average(ref_pt23, top_lip[6]), average(ref_pt19, chin[10])], np.int32)
    # pts = interpolate_pts_y(pts, on=False)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)
    # pts = np.array([average(ref_pt20, nose_tip[0]), average(ref_pt22, top_lip[0]), average(ref_pt18, chin[6])], np.int32)

    pts = np.array([average(ref_pt18, chin[6]), average(ref_pt22, top_lip[0]), average(ref_pt20, nose_tip[0])], np.int32)
    if 'left' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt21, nose_tip[4]), average(ref_pt23, top_lip[6]), average(ref_pt19, chin[10])], np.int32)
    if 'right' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt24 = np.average([nose_bridge[1], nose_bridge[2]], weights=[1, 1.5], axis=0)
    ref_pt25 = np.average([chin[2], nose_bridge[2]], weights=[1, 1.5], axis=0)
    ref_pt26 = np.average([chin[14], nose_bridge[2]], weights=[1, 1.5], axis=0)
    ref_pt27 = np.average([chin[3], nose_bridge[2]], weights=[1.6, 1], axis=0)
    ref_pt28 = np.average([chin[13], nose_bridge[2]], weights=[1.6, 1], axis=0)
    ref_pt29 = np.average([chin[4], chin[5]], weights=[1, 1.5], axis=0)
    ref_pt30 = np.average([chin[11], chin[12]], weights=[1.5, 1], axis=0)

    #pts = np.array([ref_pt24, ref_pt25, ref_pt27, ref_pt29], np.int32)
    pts = np.array([ref_pt29, ref_pt27, ref_pt25, ref_pt24], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt24, ref_pt26, ref_pt28, ref_pt30], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([chin[5], average(ref_pt27, ref_pt22), average(ref_pt25, ref_pt20, weights=[1, 2]), average(ref_pt24, nose_bridge[2])], np.int32)
    if 'left' in side:
        pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt24, nose_bridge[2]), average(ref_pt26, ref_pt21, weights=[1, 2]), average(ref_pt28, ref_pt23), chin[11]], np.int32)
    if 'right' in side:
        pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)


    # pts= np.array([chin[5], average(nose_bridge[2], chin[4], weights=[1.5, 1]), nose_bridge[2] ], np.int32)
    # pts = interpolate_pts(pts, on=True)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

   # pts= np.array([chin[11], average(nose_bridge[2], chin[13], weights=[1.5, 1]), nose_bridge[2]  ], np.int32)
   # pts = interpolate_pts(pts, on=True)
   # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt31 = np.average([nose_bridge[1], nose_bridge[2]], weights=[3, 1], axis=0)
    ref_pt32 = np.average([chin[2], nose_bridge[1]], weights=[1, 1.15], axis=0)
    ref_pt33 = np.average([chin[14], nose_bridge[1]], weights=[1, 1.15], axis=0)
    ref_pt34 = np.average([chin[3], nose_bridge[0]], weights=[4, 1], axis=0)
    ref_pt35 = np.average([chin[13], nose_bridge[0]], weights=[4, 1], axis=0)
    ref_pt36 = np.average([chin[3], chin[4]], weights=[1, 1.5], axis=0)
    ref_pt37 = np.average([chin[12], chin[13]], weights=[1.5, 1], axis=0)

    pts = np.array([ref_pt36, ref_pt34, ref_pt32, ref_pt31], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt31, ref_pt33, ref_pt35, ref_pt37], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([chin[4], average(ref_pt34, ref_pt27), average(ref_pt32, ref_pt25), average(ref_pt31, ref_pt24)], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt31, ref_pt24), average(ref_pt33, ref_pt26), average(ref_pt35, ref_pt28), chin[12]], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt38 = np.average([nose_bridge[0], nose_bridge[1]], weights=[1, 2], axis=0)
    ref_pt39 = np.average([chin[2], nose_bridge[0]], weights=[1, 1.15], axis=0)
    ref_pt40 = np.average([chin[14], nose_bridge[0]], weights=[1, 1.15], axis=0)
    ref_pt41 = np.average([chin[2], nose_bridge[1]], weights=[4, 1], axis=0)
    ref_pt42 = np.average([chin[14], nose_bridge[1]], weights=[4, 1], axis=0)
    ref_pt43 = np.average([chin[2], chin[3]], weights=[1, 2], axis=0)
    ref_pt44 = np.average([chin[13], chin[14]], weights=[2, 1], axis=0)

    pts = np.array([ref_pt43, ref_pt41, ref_pt39, ref_pt38], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt38, ref_pt40, ref_pt42, ref_pt44], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([chin[3], average(ref_pt41, ref_pt34), average(ref_pt39, ref_pt32), average(ref_pt38, ref_pt31)], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt38, ref_pt31), average(ref_pt40, ref_pt33), average(ref_pt42, ref_pt35), chin[13]], np.int32)
    pts = interpolate_pts(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt45 = np.average([nose_bridge[0], nose_bridge[1]], weights=[2, 1], axis=0)
    ref_pt46 = np.average([left_eye[3], nose_bridge[1]], weights=[1, 1], axis=0)
    ref_pt47 = np.average([right_eye[0], nose_bridge[1]], weights=[1, 1], axis=0)
    ref_pt48 = np.average([(np.average([chin[2], chin[1]], axis=0)), nose_bridge[0]], weights=[1, 1.5], axis=0)
    ref_pt49 = np.average([(np.average([chin[14], chin[15]], axis=0)), nose_bridge[0]], weights=[1, 1.5], axis=0)
    ref_pt50 = np.average([chin[2], left_eye[5]], weights=[1, 2], axis=0)
    ref_pt51 = np.average([chin[14], right_eye[4]], weights=[1, 2], axis=0)
    ref_pt52 = np.average([chin[1], chin[2]], weights=[1, 2], axis=0)
    ref_pt53 = np.average([chin[14], chin[15]], weights=[2, 1], axis=0)

    pts = np.array([ref_pt52, average(average(chin[1], chin[2], weights=[4, 1]), left_eye[5]), #ref_pt50,
                    ref_pt48, ref_pt46, ref_pt45], np.int32)
    pts = interpolate_pts_x(pts, on=False)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([ref_pt45, ref_pt47, ref_pt49, #ref_pt51,
                    average(average(chin[15], chin[14], weights=[4, 1]), right_eye[4]), ref_pt53], np.int32)
    pts = interpolate_pts_x(pts, on=False)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt52, ref_pt43), average(average(average(chin[1], chin[2], weights=[4, 1]),
                                                                 left_eye[5]), ref_pt41), average(ref_pt39, ref_pt48)], np.int32)
    pts = interpolate_pts_x(pts, on=False)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(ref_pt49, ref_pt40), average(average(average(chin[15], chin[14], weights=[4, 1]),
                                                                 right_eye[4]), ref_pt42), average(ref_pt53, ref_pt44)], np.int32)
    #pts = interpolate_pts_x(pts, kind='linear', range=int(pts[-1, 0] - pts[0, 0]) / 10)
    pts= interpolate_pts_x(pts, on=False)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)


    # ref_pt54 = np.average([chin[1], left_eye[0]], weights=[1, 1.15], axis=0)
    # ref_pt55 = np.average([chin[15], right_eye[3]], weights=[1, 1.15], axis=0)

    # pts = np.array([average(chin[1], chin[2], weights=[1, 2]), ref_pt54, left_eye[5]], np.int32)
    # pts = interpolate_pts_x(pts, on=True)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)
    #
    # pts = np.array([right_eye[4], ref_pt55,  average(chin[15], chin[14], weights=[1, 2])], np.int32)
    # pts = interpolate_pts_x(pts, on=True)
    # im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)


    ref_pt58 = np.average([chin[0], chin[1]], weights=[2, 1], axis=0)
    ref_pt59 = np.average([chin[16], chin[15]], weights=[2, 1], axis=0)

    ref_pt56 = np.average([ref_pt58, left_eye[0]], weights=[1, 1.5], axis=0)
    ref_pt57 = np.average([ref_pt59, right_eye[3]], weights=[1, 1.5], axis=0)

    pts = np.array([average(chin[0], chin[1], weights=[1, 1]), ref_pt56, left_eye[0]], np.int32)
    pts = interpolate_pts_x(pts, on=False)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], ref_pt57, average(chin[15], chin[16], weights=[1, 1])], np.int32)
    pts = interpolate_pts_x(pts, on=False)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)


    pts = np.array([chin[0], ref_pt56, left_eye[0]], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], ref_pt57, chin[16]], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    ref_pt60 = np.average([chin[0], chin[1]], weights=[1, 2], axis=0)
    ref_pt61 = np.average([chin[16], chin[15]], weights=[1, 2], axis=0)

    ref_pt62 = np.average([ref_pt60, left_eye[0]], weights=[1, 1.5], axis=0)
    ref_pt63 = np.average([ref_pt61, right_eye[3]], weights=[1, 1.5], axis=0)

    pts = np.array([chin[1], ref_pt62, left_eye[0]], np.int32)
    if 'left' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], ref_pt63, chin[15]], np.int32)
    if 'right' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[1], chin[2]), ref_pt62, left_eye[0]], np.int32)
    if 'left' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], ref_pt63, average(chin[15], chin[14])], np.int32)
    if 'right' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[0], left_eyebrow[0]), average(chin[0], left_eye[0], weights=[1, 3]), left_eye[0]], np.int32)
    if 'left' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([right_eye[3], average(chin[16], right_eye[3], weights=[1, 3]), average(chin[16], right_eyebrow[4])], np.int32)
    if 'right' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    #circle eye middle
    pts = np.array([average(chin[0], left_eye[0], weights=[1, 4]), average(left_eyebrow[2], left_eye[1]),
                    average(left_eyebrow[3], left_eye[1]), average(left_eye[3], right_eye[0], weights=[4, 1])],  np.int32)
    if 'left' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[0], left_eye[0], weights=[1, 4]), average(average(ref_pt39, ref_pt48), left_eye[5]),
                    average(average(ref_pt39, ref_pt48), left_eye[4]),
                    average(left_eye[3], right_eye[0], weights=[4, 1])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(left_eye[3], right_eye[0], weights=[1, 4]), average(right_eyebrow[1], right_eye[1], weights=[1,1]),
                    average(right_eyebrow[2], right_eye[2]), average(chin[16], right_eye[3], weights=[1, 4])],  np.int32)
    if 'right' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(left_eye[3], right_eye[0], weights=[1, 4]), average(average(ref_pt49, ref_pt40), right_eye[5]),
                    average(average(ref_pt49, ref_pt40), right_eye[4]), average(chin[16], right_eye[3], weights=[1, 4])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    #circle eye close
    pts = np.array([average(chin[0], left_eye[0], weights=[1,8]), average(left_eyebrow[2], left_eye[1], weights=[1,3]),
                    average(left_eyebrow[3], left_eye[1], weights=[1,3]), average(left_eye[3], right_eye[0], weights=[8, 1])],
                   np.int32)
    if 'left' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[0], left_eye[0], weights=[1, 8]), average(average(ref_pt39, ref_pt48), left_eye[5], weights=[1, 4]),
                    average(average(ref_pt39, ref_pt48), left_eye[4], weights=[1, 4]),
                    average(left_eye[3], right_eye[0], weights=[6, 1])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 8]), average(right_eyebrow[1], right_eye[1], weights=[1, 4]),
         average(right_eyebrow[2], right_eye[2], weights=[1, 4]), average(chin[16], right_eye[3], weights=[1, 8])], np.int32)
    if 'right' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 8]), average(average(ref_pt49, ref_pt40), right_eye[5], weights=[1,3]),
         average(average(ref_pt49, ref_pt40), right_eye[4], weights=[1,3]), average(chin[16], right_eye[3], weights=[1, 8])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    #circle eye far
    pts = np.array(
        [average(chin[0], left_eye[0], weights=[1, 2]), average(left_eyebrow[2], left_eye[1], weights=[3, 1]),
         average(left_eyebrow[3], left_eye[1], weights=[3, 1]), average(left_eye[3], right_eye[0], weights=[2, 1])],
        np.int32)
    if 'left' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array([average(chin[0], left_eye[0], weights=[1, 2]),
                    average(ref_pt39, left_eye[5], weights=[4, 1]),
                    average(ref_pt39, left_eye[4], weights=[4, 1]),
                    average(left_eye[3], right_eye[0], weights=[2, 1])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 2]), average(right_eyebrow[1], right_eye[1], weights=[3, 1]),
         average(right_eyebrow[2], right_eye[2], weights=[3, 1]), average(chin[16], right_eye[3], weights=[1, 2])], np.int32)
    if 'right' in side:
        pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = np.array(
        [average(left_eye[3], right_eye[0], weights=[1, 2]), average(ref_pt40, right_eye[5], weights=[4, 1]),
         average(ref_pt40, right_eye[4], weights=[4,1]), average(chin[16], right_eye[3], weights=[1, 2])], np.int32)
    pts = interpolate_pts_x(pts, on=True)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)


    boundary_upper = face[1]
    boundary_lower = left_eyebrow[3]
    boundary_lower = boundary_lower[1]
    diff = boundary_lower - boundary_upper

    pts_lower = np.array([left_eyebrow[0], left_eyebrow[1], left_eyebrow[3],
                          right_eyebrow[1], right_eyebrow[3], right_eyebrow[4]], np.int32)

    #pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 5 * 2
    pts = interpolate_pts_x(pts, on=True, kind='linear', range= int(pts[-1, 0] - pts[0, 0])/10)
    pts = interpolate_pts_x(pts, on=True)

    if 'left' in side:
        pts = np.concatenate((np.array([chin[0]]), pts), axis=0)
    elif 'right' in side:
        pts = np.concatenate((pts, np.array([chin[16]])), axis=0)
    else:
        pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 5 * 1
    pts = interpolate_pts_x(pts, on=True, kind='linear', range=int(pts[-1, 0] - pts[0, 0]) / 10)
    pts = interpolate_pts_x(pts, on=True)
    if 'left' in side:
        pts = np.concatenate((np.array([chin[0]]), pts), axis=0)
    elif 'right' in side:
        pts = np.concatenate((pts, np.array([chin[16]])), axis=0)
    else:
        pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 5 * 3
    pts = interpolate_pts_x(pts, on=True, kind='linear', range=int(pts[-1, 0] - pts[0, 0]) / 10)
    pts = interpolate_pts_x(pts, on=True)
    if 'left' in side:
        pts = np.concatenate((np.array([chin[0]]), pts), axis=0)
    elif 'right' in side:
        pts = np.concatenate((pts, np.array([chin[16]])), axis=0)
    else:
        pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 4 * 3
    pts = interpolate_pts_x(pts, on=True, kind='linear', range=int(pts[-1, 0] - pts[0, 0]) / 10)
    pts = interpolate_pts_x(pts, on=True)
    if 'left' in side:
        pts = np.concatenate((np.array([chin[0]]), pts), axis=0)
    elif 'right' in side:
        pts = np.concatenate((pts, np.array([chin[16]])), axis=0)
    else:
        pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)


    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 10 * 3
    pts = interpolate_pts_x(pts, on=True, kind='linear', range=int(pts[-1, 0] - pts[0, 0]) / 10)
    pts = interpolate_pts_x(pts, on=True)
    if 'left' in side:
        pts = np.concatenate((np.array([chin[0]]), pts), axis=0)
    elif 'right' in side:
        pts = np.concatenate((pts, np.array([chin[16]])), axis=0)
    else:
        pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 2 * 1
    pts = interpolate_pts_x(pts, on=True, kind='linear', range=int(pts[-1, 0] - pts[0, 0]) / 10)
    pts = interpolate_pts_x(pts, on=True)
    if 'left' in side:
        pts = np.concatenate((np.array([chin[0]]), pts), axis=0)
    elif 'right' in side:
        pts = np.concatenate((pts, np.array([chin[16]])), axis=0)
    else:
        pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    pts = pts_lower.copy()
    pts[:, 1] = pts[:, 1] - diff / 10 * 1
    pts = interpolate_pts_x(pts, on=True, kind='linear', range=int(pts[-1, 0] - pts[0, 0]) / 10)
    pts = interpolate_pts_x(pts, on=True)
    if 'left' in side:
        pts = np.concatenate((np.array([chin[0]]), pts), axis=0)
    elif 'right' in side:
        pts = np.concatenate((pts, np.array([chin[16]])), axis=0)
    else:
        pts = np.concatenate((np.array([chin[0]]), pts, np.array([chin[16]])), axis=0)

    pts = pts.astype(int)
    im = cv2.polylines(im, [pts], isClosed=False, color=color, thickness=thickness)

    return im


def draw_lines_frontal_file(file):
    img = cv2.imread(file)
    img = draw_lines_frontal(img)
    cv2.imwrite(file, img)

def draw_lines_side_file(file):
    img = cv2.imread(file)
    img = draw_lines_semifrontal(img)
    cv2.imwrite(file, img)