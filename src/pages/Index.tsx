import { Camera, MessageSquare, Leaf, Shield, Zap, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { NavLink } from "../components/NavLink"; // FIXED: Using relative path to resolve build issues

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="text-2xl font-bold text-primary">AgriDetect</div>
          <div className="flex gap-6">
            <NavLink to="/" className="text-primary font-semibold">
              Home
            </NavLink>
            <NavLink to="/detection" className="text-foreground hover:text-primary transition-colors">
              Detection
            </NavLink>
            <NavLink to="/chat" className="text-foreground hover:text-primary transition-colors">
              AI Assistant
            </NavLink>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden bg-hero-gradient text-primary-foreground">
        <div className="container mx-auto px-4 py-24 md:py-32">
          <div className="max-w-3xl mx-auto text-center space-y-6">
            <div className="inline-block mb-4">
              <Leaf className="w-16 h-16 mx-auto animate-pulse" />
            </div>
            <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
              Smart Plant Disease Detection
            </h1>
            <p className="text-xl md:text-2xl opacity-90">
              Protect your crops with AI-powered disease detection and expert farming assistance
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-6">
              <NavLink to="/detection">
                <Button size="lg" className="bg-accent hover:bg-accent/90 text-accent-foreground shadow-lg">
                  <Camera className="w-5 h-5 mr-2" />
                  Start Detection
                </Button>
              </NavLink>
              <NavLink to="/chat">
                <Button size="lg" variant="outline" className="bg-card/20 hover:bg-card/40 border-primary-foreground/30 text-primary-foreground">
                  <MessageSquare className="w-5 h-5 mr-2" />
                  Ask AI Assistant
                </Button>
              </NavLink>
            </div>
          </div>
        </div>
        <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-background to-transparent" />
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gradient-to-b from-background to-muted">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-foreground mb-4">
              Powerful Features for Modern Farming
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Everything you need to protect and optimize your agricultural operations
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <Card className="p-8 text-center hover:shadow-card-hover transition-all duration-300 border-2 hover:border-primary/50">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <Camera className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-2xl font-bold text-foreground mb-4">Live Detection</h3>
              <p className="text-muted-foreground leading-relaxed">
                Use your camera for real-time plant disease identification with advanced AI models trained on thousands of plant diseases
              </p>
            </Card>

            <Card className="p-8 text-center hover:shadow-card-hover transition-all duration-300 border-2 hover:border-primary/50">
              <div className="w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <MessageSquare className="w-8 h-8 text-accent" />
              </div>
              <h3 className="text-2xl font-bold text-foreground mb-4">AI Assistant</h3>
              <p className="text-muted-foreground leading-relaxed">
                Get instant expert advice on diseases, treatments, and general agriculture questions from our intelligent chatbot
              </p>
            </Card>

            <Card className="p-8 text-center hover:shadow-card-hover transition-all duration-300 border-2 hover:border-primary/50">
              <div className="w-16 h-16 bg-secondary/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <Shield className="w-8 h-8 text-secondary" />
              </div>
              <h3 className="text-2xl font-bold text-foreground mb-4">Treatment Plans</h3>
              <p className="text-muted-foreground leading-relaxed">
                Receive detailed treatment recommendations and prevention strategies tailored to your specific crop issues
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20 bg-muted">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-foreground mb-4">
                Why Choose AgriDetect?
              </h2>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {[
                { icon: Zap, title: "Instant Results", desc: "Get disease identification in seconds, not days" },
                { icon: CheckCircle, title: "High Accuracy", desc: "Advanced AI trained on extensive agricultural datasets" },
                { icon: Shield, title: "Preventive Care", desc: "Early detection helps prevent crop loss" },
                { icon: MessageSquare, title: "24/7 Support", desc: "AI assistant available anytime you need help" },
              ].map((benefit, index) => (
                <Card key={index} className="p-6 flex gap-4 hover:shadow-card-hover transition-shadow">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <benefit.icon className="w-6 h-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-foreground mb-2">{benefit.title}</h3>
                    <p className="text-muted-foreground">{benefit.desc}</p>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-br from-primary to-primary/80 text-primary-foreground">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl font-bold mb-6">
            Ready to Protect Your Crops?
          </h2>
          <p className="text-xl mb-8 opacity-90 max-w-2xl mx-auto">
            Start detecting plant diseases and get expert AI assistance today
          </p>
          <NavLink to="/detection">
            <Button size="lg" className="bg-accent hover:bg-accent/90 text-accent-foreground shadow-xl">
              <Camera className="w-5 h-5 mr-2" />
              Start Free Detection
            </Button>
          </NavLink>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-card border-t border-border py-8">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>&copy; 2024 AgriDetect. Empowering farmers with AI technology.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
```eof