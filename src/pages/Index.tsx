import { Camera, MessageSquare, Leaf, Shield, Zap, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { NavLink } from "@/components/NavLink";
import { ParticleBackground } from "@/components/ParticleBackground";
import { ParallaxSection } from "@/components/ParallaxSection";

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
      <section className="relative overflow-hidden bg-hero-gradient text-primary-foreground" style={{ perspective: '1000px' }}>
        <ParticleBackground />
        <div className="absolute inset-0 bg-glow-gradient animate-glow opacity-30" />
        <div className="container mx-auto px-4 py-24 md:py-32 relative z-10">
          <ParallaxSection speed={0.08}>
            <div className="max-w-3xl mx-auto text-center space-y-6">
              <div className="inline-block mb-4">
                <div className="relative">
                  <div className="absolute inset-0 bg-primary-foreground/20 blur-xl rounded-full" />
                  <Leaf className="w-16 h-16 mx-auto relative z-10 drop-shadow-glow" />
                </div>
              </div>
              <h1 className="text-5xl md:text-6xl font-bold tracking-tight transform hover:scale-105 transition-transform duration-300">
                Smart Plant Disease Detection
              </h1>
              <p className="text-xl md:text-2xl opacity-90 transform hover:scale-105 transition-transform duration-300">
                Protect your crops with AI-powered disease detection and expert farming assistance
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center pt-6">
                <NavLink to="/detection">
                  <Button size="lg" className="bg-accent hover:bg-accent/90 text-accent-foreground shadow-card-3d hover:shadow-glow transform hover:scale-110 hover:-translate-y-1 transition-all duration-300">
                    <Camera className="w-5 h-5 mr-2" />
                    Start Detection
                  </Button>
                </NavLink>
                <NavLink to="/chat">
                  <Button size="lg" variant="outline" className="bg-card/20 hover:bg-card/40 border-primary-foreground/30 text-primary-foreground shadow-card hover:shadow-card-hover transform hover:scale-110 hover:-translate-y-1 transition-all duration-300">
                    <MessageSquare className="w-5 h-5 mr-2" />
                    Ask AI Assistant
                  </Button>
                </NavLink>
              </div>
            </div>
          </ParallaxSection>
        </div>
        <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-background to-transparent" />
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gradient-to-b from-background to-muted relative">
        <div className="container mx-auto px-4">
          <ParallaxSection speed={0.05}>
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-foreground mb-4">
                Powerful Features for Modern Farming
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Everything you need to protect and optimize your agricultural operations
              </p>
            </div>
          </ParallaxSection>

          <ParallaxSection speed={0.08}>
            <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto" style={{ perspective: '1500px' }}>
            <Card className="p-8 text-center shadow-card-3d hover:shadow-glow transition-all duration-500 border-2 hover:border-primary/50 transform hover:scale-105 hover:-translate-y-2 hover:rotate-y-3 group">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:animate-float shadow-card group-hover:shadow-glow transition-all duration-300">
                <Camera className="w-8 h-8 text-primary group-hover:scale-110 transition-transform" />
              </div>
              <h3 className="text-2xl font-bold text-foreground mb-4 group-hover:text-primary transition-colors">Live Detection</h3>
              <p className="text-muted-foreground leading-relaxed">
                Use your camera for real-time plant disease identification with advanced AI models trained on thousands of plant diseases
              </p>
            </Card>

            <Card className="p-8 text-center shadow-card-3d hover:shadow-glow transition-all duration-500 border-2 hover:border-accent/50 transform hover:scale-105 hover:-translate-y-2 hover:rotate-y-3 group animate-float-slow">
              <div className="w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:animate-float shadow-card group-hover:shadow-glow transition-all duration-300">
                <MessageSquare className="w-8 h-8 text-accent group-hover:scale-110 transition-transform" />
              </div>
              <h3 className="text-2xl font-bold text-foreground mb-4 group-hover:text-accent transition-colors">AI Assistant</h3>
              <p className="text-muted-foreground leading-relaxed">
                Get instant expert advice on diseases, treatments, and general agriculture questions from our intelligent chatbot
              </p>
            </Card>

            <Card className="p-8 text-center shadow-card-3d hover:shadow-glow transition-all duration-500 border-2 hover:border-secondary/50 transform hover:scale-105 hover:-translate-y-2 hover:rotate-y-3 group">
              <div className="w-16 h-16 bg-secondary/10 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:animate-float shadow-card group-hover:shadow-glow transition-all duration-300">
                <Shield className="w-8 h-8 text-secondary group-hover:scale-110 transition-transform" />
              </div>
              <h3 className="text-2xl font-bold text-foreground mb-4 group-hover:text-secondary transition-colors">Treatment Plans</h3>
              <p className="text-muted-foreground leading-relaxed">
                Receive detailed treatment recommendations and prevention strategies tailored to your specific crop issues
              </p>
            </Card>
            </div>
          </ParallaxSection>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20 bg-muted relative">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <ParallaxSection speed={0.25}>
              <div className="text-center mb-12">
                <h2 className="text-4xl font-bold text-foreground mb-4">
                  Why Choose AgriDetect?
                </h2>
              </div>
            </ParallaxSection>

            <ParallaxSection speed={0.35}>
              <div className="grid md:grid-cols-2 gap-6">
              {[
                { icon: Zap, title: "Instant Results", desc: "Get disease identification in seconds, not days" },
                { icon: CheckCircle, title: "High Accuracy", desc: "Advanced AI trained on extensive agricultural datasets" },
                { icon: Shield, title: "Preventive Care", desc: "Early detection helps prevent crop loss" },
                { icon: MessageSquare, title: "24/7 Support", desc: "AI assistant available anytime you need help" },
              ].map((benefit, index) => (
                <Card key={index} className="p-6 flex gap-4 shadow-card-3d hover:shadow-glow transition-all duration-500 transform hover:scale-105 hover:-translate-y-1 group">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0 group-hover:animate-float shadow-card group-hover:shadow-card-hover transition-all">
                    <benefit.icon className="w-6 h-6 text-primary group-hover:scale-110 transition-transform" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-foreground mb-2 group-hover:text-primary transition-colors">{benefit.title}</h3>
                    <p className="text-muted-foreground">{benefit.desc}</p>
                  </div>
                </Card>
              ))}
              </div>
            </ParallaxSection>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-br from-primary to-primary/80 text-primary-foreground relative overflow-hidden" style={{ perspective: '1000px' }}>
        <ParticleBackground />
        <div className="absolute inset-0 bg-glow-gradient opacity-20 animate-glow" />
        <div className="container mx-auto px-4 text-center relative z-10">
          <ParallaxSection speed={0.05}>
            <h2 className="text-4xl font-bold mb-6 transform hover:scale-105 transition-transform duration-300">
              Ready to Protect Your Crops?
            </h2>
            <p className="text-xl mb-8 opacity-90 max-w-2xl mx-auto transform hover:scale-105 transition-transform duration-300">
              Start detecting plant diseases and get expert AI assistance today
            </p>
            <NavLink to="/detection">
              <Button size="lg" className="bg-accent hover:bg-accent/90 text-accent-foreground shadow-card-3d hover:shadow-glow transform hover:scale-110 hover:-translate-y-2 transition-all duration-300">
                <Camera className="w-5 h-5 mr-2" />
                Start Free Detection
              </Button>
            </NavLink>
          </ParallaxSection>
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
