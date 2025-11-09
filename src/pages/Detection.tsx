import { useState, useRef, useEffect, useCallback } from "react";
import { Camera, AlertCircle, Upload, Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { NavLink } from "@/components/NavLink";
import { Progress } from "@/components/ui/progress";

// Define the expected structure for prediction results
interface Prediction {
    prediction: string;
    confidence: number;
    top3: { class: string; confidence: number }[];
}

const Detection = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasDetection, setHasDetection] = useState(false);
  const [detectionResult, setDetectionResult] = useState<Prediction | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();
  
  // Flask API endpoint for ML prediction
  const API_URL = "http://localhost:5000/api/predict";

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsDetecting(false);
    }
  }, []);

  const startCamera = async () => {
    stopCamera();
    try {
      // Request environment-facing camera for plant scanning
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment" } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play(); // Ensure playback starts
        setIsDetecting(true);
        setHasDetection(false);
        setDetectionResult(null);
        toast({
          title: "Camera Started",
          description: "Point your camera at the plant to detect diseases",
        });
      }
    } catch (error) {
      toast({
        title: "Camera Error",
        description: "Unable to access camera. Please check permissions.",
        variant: "destructive",
      });
      console.error("Camera access error:", error);
    }
  };

  const processAndPredict = async (imageFile: File) => {
    setIsProcessing(true);
    setUploadProgress(0);
    setHasDetection(false);
    
    const formData = new FormData();
    formData.append('image', imageFile);

    try {
        // Simulate progress bar as detection usually takes a few moments
        let progress = 0;
        const interval = setInterval(() => {
            progress = Math.min(progress + 5, 90);
            setUploadProgress(progress);
        }, 200);


        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
        });

        clearInterval(interval);
        setUploadProgress(100);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: `Server returned status ${response.status}` }));
            throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
        }

        const data: { success: boolean, error?: string, prediction: string, confidence: number, top3: { class: string; confidence: number }[] } = await response.json();
        
        if (data.success) {
            setDetectionResult(data);
            setHasDetection(true);
            toast({
                title: "Detection Complete",
                description: `Disease identified: ${data.prediction}`,
            });
        } else {
            throw new Error(data.error || "Prediction failed on the server.");
        }

    } catch (error) {
        toast({
            title: "Detection Failed",
            description: `Error: ${(error as Error).message}. Check the backend server console.`,
            variant: "destructive",
        });
        console.error("Prediction error:", error);
    } finally {
        setIsProcessing(false);
        setUploadProgress(0);
    }
  }

  const captureAndDetect = () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return;
    
    // Stop the camera stream to free up resources
    stopCamera();
    
    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Set canvas dimensions to match video frame
    canvas.width = video.videoWidth > 0 ? video.videoWidth : 640;
    canvas.height = video.videoHeight > 0 ? video.videoHeight : 480;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw the current video frame onto the canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas image to Blob (JPEG for efficiency)
    canvas.toBlob((blob) => {
        if (blob) {
            // Create a File object from the Blob
            const capturedFile = new File([blob], "captured_image.jpeg", { type: "image/jpeg" });
            processAndPredict(capturedFile);
        } else {
            toast({
                title: "Capture Error",
                description: "Could not capture image from video stream.",
                variant: "destructive",
            });
            setIsProcessing(false);
        }
    }, 'image/jpeg', 0.9);
  };
  
  // --- THIS FUNCTION IS NOW FIXED ---
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        stopCamera();
        
        // Temporarily display the uploaded image in the video placeholder
        const imgUrl = URL.createObjectURL(file);
        if (videoRef.current) {
            videoRef.current.srcObject = null;
            videoRef.current.src = imgUrl;
            videoRef.current.load();
            
            // FIX: Call predict immediately. 
            // We don't wait for the unreliable 'onloadeddata' event.
            processAndPredict(file);
        }
        
      } else {
        toast({
          title: "Invalid File Type",
          description: "Please select an image file.",
          variant: "destructive",
        });
      }
    }
  };
  
  const resetDetection = () => {
    stopCamera();
    setHasDetection(false);
    setDetectionResult(null);
    if (videoRef.current) {
        videoRef.current.src = "";
    }
  };

  const resultCard = detectionResult ? (
    <Card className="p-6 bg-accent/10 border-accent/30 shadow-lg mt-8">
        <div className="flex items-center gap-4 mb-4">
            <AlertCircle className="w-8 h-8 text-destructive flex-shrink-0" />
            <div>
                <h3 className="text-xl font-bold text-foreground">Disease Detected: {detectionResult.prediction.replace(/_/g, ' ')}</h3>
                <p className="text-muted-foreground">Confidence: <span className="font-semibold text-primary">{Math.round(detectionResult.confidence * 100)}%</span></p>
            </div>
        </div>
        
        <h4 className="font-semibold mb-2 text-primary">Top Predictions:</h4>
        <div className="space-y-1 text-sm text-foreground">
            {detectionResult.top3.map((res, index) => (
                <div key={index} className="flex justify-between">
                    <span>{res.class.replace(/_/g, ' ')}</span>
                    <span className="font-mono">{Math.round(res.confidence * 100)}%</span>
                </div>
            ))}
        </div>
        
        <NavLink to={`/chat?disease=${detectionResult.prediction.toLowerCase().replace(/ /g, '-').replace(/_/g, '-')}`} className="block mt-6">
            <Button className="w-full bg-primary hover:bg-primary/90 text-primary-foreground">
                Get AI Treatment Advice
            </Button>
        </NavLink>
    </Card>
  ) : null;

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      <nav className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <NavLink to="/" className="text-2xl font-bold text-primary">
            AgriDetect
          </NavLink>
          <div className="flex gap-4">
            <NavLink to="/" className="text-foreground hover:text-primary transition-colors">
              Home
            </NavLink>
            <NavLink to="/detection" className="text-primary font-semibold">
              Detection
            </NavLink>
            <NavLink to="/chat" className="text-foreground hover:text-primary transition-colors">
              AI Assistant
            </NavLink>
          </div>
        </div>
      </nav>

      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground">
              Plant Disease Detection
            </h1>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Use your camera or upload an image to instantly identify plant diseases
            </p>
          </div>

          <Card className="p-8 shadow-card hover:shadow-card-hover transition-shadow">
            <div className="space-y-6">
              <div className="aspect-video bg-muted rounded-lg overflow-hidden relative">
                
                {/* Canvas is used to grab the frame/image data, kept hidden */}
                <canvas ref={canvasRef} className="absolute w-full h-full object-cover hidden" />

                {isProcessing ? (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-primary bg-background/90 z-10">
                            <Loader2 className="w-10 h-10 mb-4 animate-spin" />
                            <p className="text-lg font-semibold">Analyzing Image...</p>
                            <Progress value={uploadProgress} className="w-1/2 mt-4 h-2 bg-muted-foreground/30" />
                        </div>
                ) : !isDetecting && !hasDetection ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground">
                    <Camera className="w-16 h-16 mb-4" />
                    <p className="text-lg">Camera preview / Image placeholder</p>
                  </div>
                ) : null}
                
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-cover ${!isDetecting && !detectionResult ? "hidden" : ""}`}
                />
                
                {hasDetection && detectionResult && (
                    <img 
                        src={canvasRef.current?.toDataURL() || videoRef.current?.src || "/public/placeholder.svg"} 
                        alt="Captured image for analysis" 
                        className="w-full h-full object-cover" 
                    />
                )}
              </div>

              <div className="flex flex-wrap gap-4 justify-center">
                {!isDetecting && !hasDetection && (
                  <>
                    <Button 
                      onClick={startCamera}
                      size="lg"
                      disabled={isProcessing}
                      className="bg-primary hover:bg-primary/90 text-primary-foreground"
                    >
                      <Camera className="w-5 h-5 mr-2" />
                      Start Camera
                    </Button>
                    <input 
                      type="file" 
                      accept="image/*" 
                      ref={fileInputRef} 
                      onChange={handleFileChange} 
                      style={{ display: 'none' }}
                    />
                    <Button 
                        onClick={() => fileInputRef.current?.click()}
                        size="lg"
                        variant="outline"
                        disabled={isProcessing}
                    >
                        <Upload className="w-5 h-5 mr-2" />
                        Upload Image
                    </Button>
                  </>
                )}

                {isDetecting && (
                  <>
                    <Button 
                      onClick={captureAndDetect}
                      size="lg"
                      disabled={isProcessing}
                      className="bg-accent hover:bg-accent/90 text-accent-foreground"
                    >
                      Capture & Detect
                    </Button>
                    <Button 
                      onClick={stopCamera}
                      size="lg"
                      disabled={isProcessing}
                      variant="outline"
                    >
                      Stop Camera
                    </Button>
                  </>
                )}

                {hasDetection && detectionResult && (
                  <Button 
                    onClick={resetDetection}
                    size="lg"
                    variant="outline"
                    disabled={isProcessing}
                  >
                    <RefreshCw className="w-5 h-5 mr-2" />
                    Scan Another Plant
                  </Button>
                )}
              </div>
            </div>
          </Card>

          {resultCard}

          <Card className="p-6 bg-muted/50 border-accent/20">
            <h3 className="text-lg font-semibold text-foreground mb-3">How it works:</h3>
            <ol className="space-y-2 text-muted-foreground">
              <li>1. Click "Start Camera" or "Upload Image" to provide a plant leaf image.</li>
              <li>2. The image is sent to the Flask AI backend for analysis.</li>
              <li>3. Our **Hybrid PlantNet (CNN + ViT)** model identifies the disease.</li>
              <li>4. Get instant results and a link to the AI Assistant for customized treatment advice.</li>
            </ol>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Detection;