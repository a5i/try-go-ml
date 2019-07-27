// based on https://github.com/hybridgroup/gocv/blob/master/cmd/dnn-pose-detection/main.go
package main

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/a5i/try-go-ml/gopher"
	"gocv.io/x/gocv"
)

var net *gocv.Net
var images chan *gocv.Mat
var poses chan [][]image.Point

func main() {
	const deviceID = 0
	const proto = "Transportation/human_pose_estimation/mobilenet-v1/dldt/human-pose-estimation-0001-int8.xml"
	const model = "Transportation/human_pose_estimation/mobilenet-v1/dldt/human-pose-estimation-0001-int8.bin"
	backend := gocv.NetBackendOpenVINO
	target := gocv.NetTargetCPU

	window := gocv.NewWindow("DNN Pose Detection")
	defer window.Close()

	windowG := gocv.NewWindow("Gopher")
	defer windowG.Close()

	// open capture device
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	img := gocv.NewMat()
	defer img.Close()

	// open OpenPose model
	n := gocv.ReadNet(model, proto)
	net = &n
	if net.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", model, proto)
		return
	}

	defer net.Close()
	net.SetPreferableBackend(gocv.NetBackendType(backend))
	net.SetPreferableTarget(gocv.NetTargetType(target))

	fmt.Printf("Start reading device: %v\n", deviceID)

	images = make(chan *gocv.Mat, 1)
	poses = make(chan [][]image.Point)

	if ok := webcam.Read(&img); !ok {
		fmt.Printf("Error cannot read device %v\n", deviceID)
		return
	}

	processFrame(&img)

	go performDetection()
	var poseState [][]image.Point
	gp := gopher.NewGopher()
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}
		select {
		case pose := <-poses:
			// we've received the next pose from channel, so send next image frame for detection
			processFrame(&img)
			poseState = pose
			//p :=
			if left := pose[3]; len(left) > 1 {
				a := math.Atan2(-float64(left[0].Y-left[1].Y), -float64(left[0].X-left[1].X))
				gp.RotateLeft(a)
			}
			if right := pose[5]; len(right) > 1 {
				a := math.Atan2(-float64(right[0].Y-right[1].Y), -float64(right[0].X-right[1].X))
				gp.RotateRight(a)
			}

			gp.Render()
		default:
			// show current frame without blocking, so do nothing here
		}

		drawPose(poseState, &img)
		window.IMShow(img)
		windowG.IMShow(gp.RenderMat)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}

func processFrame(i *gocv.Mat) {
	frame := gocv.NewMat()
	i.CopyTo(&frame)
	images <- &frame
}

// performDetection analyzes the results from the detector network.
// the result is an array of "heatmaps" which are the probability
// of a body part being in location x,y
func performDetection() {
	for {
		// get next frame from channel
		frame := <-images

		blob := gocv.BlobFromImage(*frame, 1.0, image.Pt(256, 256), gocv.NewScalar(0, 0, 0, 0), false, false)

		// feed the blob into the detector
		net.SetInput(blob, "data")

		// run a forward pass thru the network
		prob := net.Forward("Mconv7_stage2_L2")

		var midx int

		s := prob.Size()
		nparts, h, w := s[1], s[2], s[3]

		// find out, which model we have
		switch nparts {
		case 19:
			// COCO body
			midx = 0
			nparts = 18 // skip background
		case 16:
			// MPI body
			midx = 1
			nparts = 15 // skip background
		case 22:
			// hand
			midx = 2
		default:
			fmt.Println("there should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand model")
			return
		}

		// find the most likely match for each part
		pts := make([]image.Point, 22)
		for i := 0; i < nparts; i++ {
			pts[i] = image.Pt(-1, -1)
			heatmap, _ := prob.FromPtr(h, w, gocv.MatTypeCV32F, 0, i)

			_, maxVal, _, maxLoc := gocv.MinMaxLoc(heatmap)

			if maxVal > 0.1 {
				pts[i] = maxLoc
			}
			heatmap.Close()
		}

		// determine scale factor
		sX := int(float32(frame.Cols()) / float32(w))
		sY := int(float32(frame.Rows()) / float32(h))

		// create the results array of pairs of points with the lines that best fit
		// each body part, e.g.
		// [[point A for body part 1, point B for body part 1],
		//  [point A for body part 2, point B for body part 2], ...]
		results := [][]image.Point{}
		for _, p := range PosePairs[midx] {
			a := pts[p[0]]
			b := pts[p[1]]

			// high enough confidence in this pose?
			if a.X <= 0 || a.Y <= 0 || b.X <= 0 || b.Y <= 0 {
				results = append(results, []image.Point{})
				continue
			}

			// scale to image size
			a.X *= sX
			a.Y *= sY
			b.X *= sX
			b.Y *= sY

			results = append(results, []image.Point{a, b})
		}
		prob.Close()
		blob.Close()
		frame.Close()

		// send pose results in channel
		poses <- results
	}
}

func drawPose(pose [][]image.Point, frame *gocv.Mat) {
	for _, pts := range pose {
		if len(pts) == 0 {
			continue
		}
		gocv.Line(frame, pts[0], pts[1], color.RGBA{0, 255, 0, 0}, 2)
		gocv.Circle(frame, pts[0], 3, color.RGBA{0, 0, 200, 0}, -1)
		gocv.Circle(frame, pts[1], 3, color.RGBA{0, 0, 200, 0}, -1)
	}
}

// PosePairs is a table of the body part connections in the format [model_id][pair_id][from/to]
// For details please see:
// https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
//
var PosePairs = [3][20][2]int{
	{ // COCO body
		{1, 2}, {1, 5}, {2, 3},
		{3, 4}, {5, 6}, {6, 7},
		{1, 8}, {8, 9}, {9, 10},
		{1, 11}, {11, 12}, {12, 13},
		{1, 0}, {0, 14},
		{14, 16}, {0, 15}, {15, 17},
	},
	{ // MPI body
		{0, 1}, {1, 2}, {2, 3},
		{3, 4}, {1, 5}, {5, 6},
		{6, 7}, {1, 14}, {14, 8}, {8, 9},
		{9, 10}, {14, 11}, {11, 12}, {12, 13},
	},
	{ // hand
		{0, 1}, {1, 2}, {2, 3}, {3, 4}, // thumb
		{0, 5}, {5, 6}, {6, 7}, {7, 8}, // pinkie
		{0, 9}, {9, 10}, {10, 11}, {11, 12}, // middle
		{0, 13}, {13, 14}, {14, 15}, {15, 16}, // ring
		{0, 17}, {17, 18}, {18, 19}, {19, 20}, // small
	}}
