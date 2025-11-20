import { useState, useRef, useCallback } from "react";
import { Camera, AlertCircle, Upload, Loader2, TrendingUp } from "lucide-react";
// NOTE: Assuming your project has defined components like Button, Card, useToast, etc.
// For demonstration, these are simplified or assumed to be available.

// Placeholder for external components/hooks needed for compilation
const Button = (props) => <button {...props} className={"p-2 rounded-lg " + props.className}>{props.children}</button>;
const Card = (props) => <div {...props} className={"bg-white p-4 rounded-xl shadow-lg " + props.className}>{props.children}</div>;
const NavLink = (props) => <a href={props.to} className={props.className}>{props.children}</a>;
const ParallaxSection = (props) => <div {...props}>{props.children}</div>;
const useToast = () => ({ toast: (options) => console.log('Toast:', options.title, options.description) });
// End placeholders

// --- TYPES FOR API RESPONSE ---
interface PredictionResult {
    prediction: string;
    confidence: string;
    top3: { class: string; confidence: string }[];
}

const Detection = () => {
    const [isDetecting, setIsDetecting] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [hasDetection, setHasDetection] = useState(false);
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);

    const videoRef = useRef<HTMLVideoElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const { toast } = useToast();

    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
            setIsDetecting(false);
        }
    };

    const startCamera = async () => {
        setHasDetection(false);
        setUploadedImage(null);
        setPredictionResult(null);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: "environment" } 
            });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setIsDetecting(true);
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
        }
    };

    // --- NEW: Function to handle API submission ---
    const runDetection = useCallback(async (file: File) => {
        setIsLoading(true);
        setHasDetection(false);
        setPredictionResult(null);
        stopCamera();

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch("http://localhost:5000/api/predict", {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.success) {
                setPredictionResult(data);
                setHasDetection(true);
                toast({ title: "Detection Complete", description: `Identified: ${data.prediction}` });
            } else {
                 toast({ title: "Prediction Failed", description: data.error, variant: "destructive" });
            }

        } catch (error) {
            console.error("API Prediction Error:", error);
            toast({
                title: "Server Error",
                description: `Failed to connect to ML API: ${error.message}`,
                variant: "destructive",
            });
        } finally {
            setIsLoading(false);
        }
    }, [toast]);
    // --- END NEW FUNCTION ---


    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                // Set image preview
                setUploadedImage(e.target?.result as string);
                // Run the actual API prediction
                runDetection(file);
            };
            reader.readAsDataURL(file);
        }
    };

    const captureAndDetect = () => {
        if (!videoRef.current) return;

        // 1. Capture image from video stream
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
            
            // 2. Convert canvas image to Blob/File
            canvas.toBlob((blob) => {
                if (blob) {
                    const capturedFile = new File([blob], "captured_plant.jpeg", { type: "image/jpeg" });
                    
                    // Set image preview (optional, for confirmation)
                    setUploadedImage(URL.createObjectURL(blob));
                    
                    // 3. Run detection on the captured file
                    runDetection(capturedFile);
                } else {
                    toast({ title: "Capture Failed", description: "Could not create image file.", variant: "destructive" });
                }
            }, 'image/jpeg');
        }
        stopCamera();
    };

    const resetDetection = () => {
        setHasDetection(false);
        setUploadedImage(null);
        setPredictionResult(null);
        setIsDetecting(false);
        setIsLoading(false);
    };


    // Helper for generating NavLinks in the fixed layout
    const NavItem = ({ to, children }) => (
        <NavLink to={to} className="text-foreground hover:text-primary transition-colors">
            {children}
        </NavLink>
    );

    return (
        <div className="min-h-screen bg-gradient-to-b from-background to-muted font-sans">
            <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
                <div className="container mx-auto px-4 py-4 flex items-center justify-between">
                    <NavLink to="/" className="text-2xl font-bold text-primary">
                        AgriDetect
                    </NavLink>
                    <div className="flex gap-4">
                        <NavItem to="/">Home</NavItem>
                        <NavLink to="/detection" className="text-primary font-semibold">
                            Detection
                        </NavLink>
                        <NavItem to="/chat">AI Assistant</NavItem>
                    </div>
                </div>
            </nav>

            <main className="container mx-auto px-4 py-12">
                <div className="max-w-4xl mx-auto space-y-8">
                    <ParallaxSection speed={0.05}>
                        <div className="text-center space-y-4">
                            <h1 className="text-4xl md:text-5xl font-bold text-foreground">
                                Plant Disease Detection
                            </h1>
                            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                                Use your camera or upload an image to instantly identify plant diseases and get AI-powered treatment recommendations
                            </p>
                        </div>
                    </ParallaxSection>

                    <ParallaxSection speed={0.08}>
                        <Card className="p-8 shadow-card-3d hover:shadow-glow transition-all duration-500 transform hover:scale-[1.02]">
                            <div className="space-y-6">
                                {/* IMAGE/VIDEO DISPLAY AREA */}
                                <div className="aspect-video bg-muted rounded-lg overflow-hidden relative shadow-card-3d" style={{ perspective: '1000px' }}>
                                    
                                    {!isDetecting && !hasDetection && !uploadedImage && !isLoading && (
                                        <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground bg-glow-gradient">
                                            <Camera className="w-16 h-16 mb-4 drop-shadow-glow" />
                                            <p className="text-lg">Camera preview or uploaded image will appear here</p>
                                        </div>
                                    )}
                                    
                                    {/* Loading State */}
                                    {isLoading && (
                                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-primary/20 backdrop-blur-sm">
                                            <Loader2 className="w-12 h-12 text-primary animate-spin mb-4" />
                                            <p className="text-primary font-semibold">Analyzing image...</p>
                                            <p className="text-sm text-primary/80">Connecting to ML server (Port 5000)</p>
                                        </div>
                                    )}

                                    {/* Video Stream */}
                                    <video
                                        ref={videoRef}
                                        autoPlay
                                        playsInline
                                        className={`w-full h-full object-cover ${!isDetecting ? "hidden" : ""}`}
                                    />

                                    {/* Uploaded Image Preview */}
                                    {uploadedImage && !isDetecting && (
                                        <img src={uploadedImage} alt="Uploaded plant" className="w-full h-full object-cover" />
                                    )}

                                    {/* Detection Result Overlay (Dynamically Rendered) */}
                                    {hasDetection && predictionResult && (
                                        <div className="absolute inset-0 bg-primary/10 backdrop-blur-sm flex flex-col items-center justify-center">
                                            <div className="bg-card p-6 rounded-lg shadow-2xl text-center max-w-md border border-accent/30 transform transition-all duration-500 hover:scale-105">
                                                <AlertCircle className="w-12 h-12 text-accent mx-auto mb-4 drop-shadow-lg" />
                                                <h3 className="text-2xl font-bold text-foreground mb-2">
                                                    Disease Identified
                                                </h3>
                                                <p className="text-xl font-semibold text-primary mb-1">
                                                    {predictionResult.prediction}
                                                </p>
                                                <p className="text-sm text-muted-foreground mb-4">
                                                    Confidence: {predictionResult.confidence}
                                                </p>
                                                
                                                {/* Top 3 Predictions */}
                                                <div className="mt-4 text-xs text-left p-2 bg-muted/50 rounded-md">
                                                    <h4 className="font-medium text-foreground flex items-center mb-1"><TrendingUp className="w-4 h-4 mr-1 text-accent" /> Top 3 Predictions:</h4>
                                                    <ul className="space-y-1">
                                                        {predictionResult.top3.map((p, index) => (
                                                            <li key={index} className="flex justify-between">
                                                                <span className="text-muted-foreground">{p.class}</span>
                                                                <span className="font-mono text-primary/80">{p.confidence}</span>
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>

                                                <NavLink to={`/chat?disease=${predictionResult.prediction.replace(/\s/g, '-')}`}>
                                                    <Button className="mt-6 bg-accent hover:bg-accent/90 text-accent-foreground w-full">
                                                        Get AI Treatment Advice
                                                    </Button>
                                                </NavLink>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* BUTTONS */}
                                <div className="flex gap-4 justify-center flex-wrap">
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        onChange={handleFileUpload}
                                        accept="image/*"
                                        className="hidden"
                                    />
                                    
                                    {/* Initial State / Post-Detection Options */}
                                    {(!isDetecting && !isLoading) && (
                                        <>
                                            <Button 
                                                onClick={startCamera}
                                                size="lg"
                                                className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-card-3d hover:shadow-glow transform hover:scale-105 hover:-translate-y-1 transition-all duration-300"
                                            >
                                                <Camera className="w-5 h-5 mr-2" />
                                                {hasDetection ? 'Use Camera' : 'Start Camera'}
                                            </Button>
                                            <Button 
                                                onClick={() => fileInputRef.current?.click()}
                                                size="lg"
                                                variant="outline"
                                                className="shadow-card hover:shadow-card-hover transform hover:scale-105 transition-all duration-300"
                                            >
                                                <Upload className="w-5 h-5 mr-2" />
                                                {hasDetection ? 'Upload Another' : 'Upload Image'}
                                            </Button>
                                        </>
                                    )}

                                    {/* Camera Active State */}
                                    {isDetecting && (
                                        <>
                                            <Button 
                                                onClick={captureAndDetect}
                                                size="lg"
                                                className="bg-accent hover:bg-accent/90 text-accent-foreground shadow-card-3d hover:shadow-glow transform hover:scale-105 hover:-translate-y-1 transition-all duration-300"
                                            >
                                                Capture & Detect
                                            </Button>
                                            <Button 
                                                onClick={stopCamera}
                                                size="lg"
                                                variant="outline"
                                                className="shadow-card hover:shadow-card-hover transform hover:scale-105 transition-all duration-300"
                                            >
                                                Stop Camera
                                            </Button>
                                        </>
                                    )}
                                </div>
                            </div>
                        </Card>
                    </ParallaxSection>

                    <ParallaxSection speed={0.05}>
                        <Card className="p-6 bg-muted/50 border-accent/20 shadow-card-3d hover:shadow-glow transition-all duration-500 transform hover:scale-[1.02]">
                            <h3 className="text-lg font-semibold text-foreground mb-3">How it works:</h3>
                            <ol className="space-y-2 text-muted-foreground">
                                <li className="transform hover:translate-x-2 transition-transform duration-300">1. Click "Start Camera" or "Upload Image" to begin</li>
                                <li className="transform hover:translate-x-2 transition-transform duration-300">2. Point the camera at the affected plant leaves or select an image</li>
                                <li className="transform hover:translate-x-2 transition-transform duration-300">3. The image is sent to the **Flask ML Server (Port 5000)** for analysis.</li>
                                <li className="transform hover:translate-x-2 transition-transform duration-300">4. Get instant results, including top predictions and confidence scores.</li>
                            </ol>
                        </Card>
                    </ParallaxSection>
                </div>
            </main>
        </div>
    );
};

export default Detection;