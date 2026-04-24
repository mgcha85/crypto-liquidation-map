package model

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type Predictor struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

func NewPredictor(modelPath string) (*Predictor, error) {
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to init ONNX runtime: %w", err)
	}

	inputShape := ort.NewShape(1, 31)
	input, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	outputShape := ort.NewShape(1, 2)
	output, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"output"},
		[]ort.ArbitraryTensor{input},
		[]ort.ArbitraryTensor{output},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Predictor{
		session: session,
		input:   input,
		output:  output,
	}, nil
}

func (p *Predictor) Predict(features []float64) (int, float64, error) {
	if len(features) != 31 {
		return 0, 0, fmt.Errorf("expected 31 features, got %d", len(features))
	}

	inputData := p.input.GetData()
	for i, f := range features {
		inputData[i] = float32(f)
	}

	err := p.session.Run()
	if err != nil {
		return 0, 0, fmt.Errorf("inference failed: %w", err)
	}

	outputData := p.output.GetData()

	prob0 := float64(outputData[0])
	prob1 := float64(outputData[1])

	if prob1 > prob0 {
		return 1, prob1, nil
	}
	return -1, prob0, nil
}

func (p *Predictor) Close() error {
	if p.session != nil {
		return p.session.Destroy()
	}
	return nil
}
