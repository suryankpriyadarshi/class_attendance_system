const toggel=document.querySelector('.toggel')
const password=document.querySelector('#Password')

toggel.addEventListener('click',()=>{
    if(password.type == 'password'){
        password.type='text'
        toggel.innerHTML= '<i class="fa fa-eye-slash"></i>' 
    }
    else
    {
        password.type='password'
        toggel.innerHTML= '<i class="fa fa-eye"></i>'
    }
}
)