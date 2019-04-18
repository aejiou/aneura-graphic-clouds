class WhackAMole {
            
    // Properties used to initialize our
    // Whack-a-Mole Game
    constructor(startButton, moles, scoreOut, gameTime, message, peepTimeMin, peepTimeMax){		
        // Game HTML Elements
        this.btnStart = startButton;
        this.moles = moles;
        this.scoreOut = scoreOut;
        this.message = message;
        
        // Game Parameters
        this.gameTime = gameTime;
        this.minTimeStart = peepTimeMin;
        this.maxTimeStart = peepTimeMax;
        this.numOfMoles = this.moles.length;
        
        // Game State Variables
        this.prevMoleNumber = null;
        this.total = 0;
        this.score = 0;
        this.gameTimer = null;
        this.peepTimer = null;	
        this.stop = false;	

        // Accuracy messages
        this.messages = [   "Probability of everything is 50%. It either happens or not.", 
                            "Let's just predict mean for everything.", 
                            "I vote for Random Forest.", 
                            "Gradient descent ready to take off.", 
                            "That's some mighty local minimum!", 
                            "Have faith in math and carry on.", 
                            "We're in too deep now.", 
                            "I have no idea how it's working.", 
                            "Time to call yourself a fitness trainer.", 
                            "Is it a miracle or are you overfitting?", 
                            "That's it. AI will now take over the world." ];
    }
    
    init(){
        this.score = 0;
        this.total = 0;
        this.stop = false;
        this.minPeepTime = this.minTimeStart;
        this.maxPeepTime = this.maxTimeStart;
        this.scoreOut.text("0.00");
        this.prevMoleNumber = null;
        this.btnStart.addClass('hidden');
        this.peep();
        this.gameTimer = setTimeout(() => {
        this.faster()
        }, 1000);
        this.message.text("Can AI take over the world?");	
    }

    faster(){
        if (this.minPeepTime>250){
        this.minPeepTime = this.minPeepTime - 5;
        this.maxPeepTime = this.maxPeepTime - 5;}
        this.gameTimer = setTimeout(() => {
        this.faster() }, 1000);

    }

    end(){
        this.gameTime.html("");
        this.btnStart.removeClass('hidden');
        this.stop = true;
    }
    
    peep(){
        const time = this._randomTime(this.minPeepTime, this.maxPeepTime);
        const mole = this._randomMole(this.moles);
        this.total++;
        this.gameTime.html("Epoch<br><em>"+this.total+"</em>");
        mole.addClass('up');
        this.peepTimer = setTimeout(() => {
            mole.removeClass('up');
            this.scoreOut.text((this.score/this.total).toFixed(2));
            if (this.total>10) {
                this.message.text(this.messages[Math.round(10*this.score/this.total-0.4)]);
                if (this.score/this.total<0.01) {
                    this.end();
                    this.message.text("No, AI won't be taking over the world any time soon.")
                }
            }
            if (!this.stop) this.peep(); 
        }, time);
    }
    
    bonk(mole) {
        mole.addClass('bonked')            
        .one('transitionend', () => {
                        mole.removeClass('up')
                        .one('transitionend', () => {
                        mole.removeClass('bonked');
                        })
        });
        this.score++;
        if (this.total>10){
            this.message.text(this.messages[Math.round(10*this.score/this.total-0.4)]);
        }
    this.scoreOut.text((this.score/this.total).toFixed(2));
    }
    
    // Utility functions
    
    // generate a random time to determine how long
    // the moles stay up
    _randomTime(min, max){
        return Math.round(Math.random() * (max - min) + min);
    }
    
    // randomly selects one of the moles to display
    _randomMole(moles) {
        const idx = Math.floor(Math.random() * this.numOfMoles);
        const mole = moles.eq(idx);
        if(idx === this.prevMoleNumber ) {
            //console.log('...same mole...try again...');
            return this._randomMole(moles);
        }
        this.prevMoleNumber = idx;
        return mole;
    }
    
}

// Get a new instance of the Whack A Mole
// class
// Parameters:
// 1. Start Button
// 2. Mole Image
// 3. Score out
// 4. Time layer
// 5. Peep Time Min (ms)
// 6. Peep Time Max (ms)
const wam = new WhackAMole($('#btn-start'), $('.mole-pic'), $('#score-out'), $('#time'),$('#message'), 1500, 2000);

// Game Event Handlers
wam.btnStart.click(function(){ wam.init(); });

wam.moles.click(function(){ wam.bonk( $(this) );

   
    //if($(this).hasClass('bonked')){
    //    return;
    //}
    
    
    
});