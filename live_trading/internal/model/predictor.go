package model

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type Predictor struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	label   *ort.Tensor[int64]
	probs   *ort.Tensor[float32]
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

	labelShape := ort.NewShape(1)
	label, err := ort.NewEmptyTensor[int64](labelShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create label tensor: %w", err)
	}

	probsShape := ort.NewShape(1, 2)
	probs, err := ort.NewEmptyTensor[float32](probsShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create probs tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"label", "probabilities"},
		[]ort.ArbitraryTensor{input},
		[]ort.ArbitraryTensor{label, probs},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Predictor{
		session: session,
		input:   input,
		label:   label,
		probs:   probs,
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

	labelData := p.label.GetData()
	probsData := p.probs.GetData()

	pred := int(labelData[0])
	confidence := float64(probsData[pred])

	if pred == 1 {
		return 1, confidence, nil
	}
	return -1, confidence, nil
}

func (p *Predictor) Close() error {
	if p.session != nil {
		return p.session.Destroy()
	}
	return nil
}
