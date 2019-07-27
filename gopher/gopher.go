package gopher

import "C"
import (
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"

	"github.com/disintegration/imaging"
	"github.com/labstack/gommon/log"
	"gocv.io/x/gocv"
)

const gopherTorso = "gopherkon/sprites/torso/cheeky/0.png"
const gopherEyes = "gopherkon/sprites/eyes/alien_fish.png"
const gopherPose = "gopherkon/sprites/pose/dunno.png"
const gopherEars = "gopherkon/sprites/ears/fancy/0.png"
const gopherNose = "gopherkon/sprites/nose/oval.png"
const gopherMouth = "gopherkon/sprites/mouth/lol.png"

type Gopher struct {
	torso        image.Image
	eyes         image.Image
	left         image.Image
	right        image.Image
	buffer       *image.RGBA
	renderBuffer *image.RGBA
	RenderMat    gocv.Mat
	leftAngel    float64
	offsetLeft   image.Point
	offsetRight  image.Point
	rightAngel   float64
}

func mustImage(img image.Image, err error) image.Image {
	if err != nil {
		log.Panic(err)
	}
	return img
}

func drawImage(dst draw.Image, src image.Image, offset image.Point) {
	r := image.Rect(
		0, 0,
		src.Bounds().Dx(),
		src.Bounds().Dy(),
	)
	draw.Draw(dst, r.Add(offset), src, src.Bounds().Min, draw.Over)
}

const sX = 490
const sY = 490

func NewGopher() *Gopher {
	g := &Gopher{}

	g.torso = mustImage(loadImage(gopherTorso))
	g.eyes = mustImage(loadImage(gopherEyes))

	g.buffer = image.NewRGBA(image.Rect(0, 0, sX, sY))
	g.renderBuffer = image.NewRGBA(image.Rect(0, 0, sX, sY))

	for i := range g.buffer.Pix {
		g.buffer.Pix[i] = 255
	}
	drawImage(g.buffer, g.torso, image.Pt(0, 0))
	drawImage(g.buffer, g.eyes, image.Pt(0, 0))
	drawImage(g.buffer, mustImage(loadImage(gopherEars)), image.Point{})
	drawImage(g.buffer, mustImage(loadImage(gopherMouth)), image.Point{})
	drawImage(g.buffer, mustImage(loadImage(gopherNose)), image.Point{})
	g.left = mustImage(loadImage("dnn-pose-detection/left.png"))
	g.right = mustImage(loadImage("dnn-pose-detection/right.png"))
	var err error
	g.RenderMat, err = gocv.NewMatFromBytes(g.renderBuffer.Bounds().Dy(), g.renderBuffer.Bounds().Dx(), gocv.MatTypeCV8UC4, g.renderBuffer.Pix)
	if err != nil {
		log.Panic(err)
	}
	return g
}

func (g *Gopher) RotateLeft(angel float64) {
	g.leftAngel = 360.0 - angel/3.14*360.0
}

func (g *Gopher) RotateRight(angel float64) {
	g.rightAngel = 360.0 - angel/3.14*360.0
}

func (g *Gopher) Render() {
	copy(g.renderBuffer.Pix, g.buffer.Pix)

	left := imaging.Rotate(g.left, g.leftAngel, color.Transparent)

	drawImage(g.renderBuffer, left, image.Pt(0, 230).Add(g.offsetLeft))
	right := imaging.Rotate(g.right, g.rightAngel, color.Transparent)
	drawImage(g.renderBuffer, right, image.Pt(sX/2+490/2-80*2, 230))
}

func loadImage(fn string) (image.Image, error) {
	f, err := os.Open(fn)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return png.Decode(f)
}
